"""Adapter store (Phase 3): dormant adapters live *outside* VRAM, ready to stream in.

Each adapter's safetensors file is loaded once into page-locked (pinned) host memory so a
later `copy_()` to the GPU is a single DMA transfer rather than a pageable copy that goes
through a staging buffer. The store is a plain dict; a Redis or DuckDB backend would only
change `_load_bytes()`, which is why the POC does not bother with a daemon.

Key normalisation: PEFT saves LoRA weights without the adapter name
    saved:  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    live:   base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
`live_to_saved_key()` strips the adapter name so the jumper can address live parameters
by the saved key.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from safetensors.torch import load_file

_LORA_KEY = re.compile(r"\.(lora_A|lora_B|lora_embedding_A|lora_embedding_B)\.([^.]+)\.")


def live_to_saved_key(name: str, adapter_name: str = "default") -> Optional[str]:
    """Map a live PEFT parameter name to the key PEFT uses in adapter_model.safetensors.
    Returns None for parameters that are not LoRA weights of `adapter_name`."""
    m = _LORA_KEY.search(name)
    if not m or m.group(2) != adapter_name:
        return None
    return name[: m.start()] + f".{m.group(1)}." + name[m.end():]


class AdapterStore:
    def __init__(self, dtype: Optional[torch.dtype] = None, pin: bool = True):
        self.dtype = dtype              # None keeps the saved dtype (fp32); torch.float16 halves bytes over PCIe
        self.pin = pin and torch.cuda.is_available()
        self._tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        self.meta: Dict[str, Dict[str, float]] = {}

    # -- loading ----------------------------------------------------------------------------
    def _load_bytes(self, path: Path) -> Dict[str, torch.Tensor]:
        return load_file(str(path))            # <- a Redis backend would deserialise here instead

    def add(self, name: str, adapter_dir) -> "AdapterStore":
        t0 = time.perf_counter()
        sd = self._load_bytes(Path(adapter_dir) / "adapter_model.safetensors")
        out = {}
        for k, v in sd.items():
            t = v.to(self.dtype) if self.dtype is not None else v
            t = t.contiguous()
            if self.pin:
                t = t.pin_memory()
            out[k] = t
        self._tensors[name] = out
        nbytes = sum(t.numel() * t.element_size() for t in out.values())
        self.meta[name] = {"tensors": len(out), "mb": nbytes / 1e6, "load_s": time.perf_counter() - t0,
                           "dtype": str(next(iter(out.values())).dtype)}
        return self

    def add_all(self, adapters: Dict[str, str]) -> "AdapterStore":
        for n, d in adapters.items():
            self.add(n, d)
        return self

    # -- access -----------------------------------------------------------------------------
    def names(self) -> Iterable[str]:
        return self._tensors.keys()

    def keys(self, name: str) -> Iterable[str]:
        return self._tensors[name].keys()

    def get(self, name: str) -> Dict[str, torch.Tensor]:
        """O(1): returns the pinned tensors. Timed separately by the jumper as 'fetch'."""
        return self._tensors[name]

    def check_same_schema(self) -> None:
        names = list(self._tensors)
        ref = {k: tuple(v.shape) for k, v in self._tensors[names[0]].items()}
        for n in names[1:]:
            cur = {k: tuple(v.shape) for k, v in self._tensors[n].items()}
            if cur != ref:
                missing = set(ref) ^ set(cur)
                raise ValueError(f"adapter '{n}' schema differs from '{names[0]}': {sorted(missing)[:5]}")

    def __repr__(self) -> str:
        return "AdapterStore(" + ", ".join(f"{n}: {m['mb']:.1f} MB {m['dtype']}" for n, m in self.meta.items()) + ")"
