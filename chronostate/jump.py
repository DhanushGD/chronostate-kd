"""Weight jump (Phase 3): swap the active adapter by overwriting the live LoRA tensors in place.

The student is loaded once with one PEFT adapter attached; that adapter is just a set of
GPU tensors in fixed slots. `WeightJumper.jump(name)` copies the stored tensors of `name`
into those slots with `Tensor.copy_()`. Nothing is allocated, no module is rebuilt, no
`load_adapter` call, and because the base weights never change the KV cache of an ongoing
conversation stays valid across the jump.

Timing is split into three phases, all wall-clock on the host:
    fetch  dict lookup in the store (would be a Redis GET + deserialise in production)
    copy   issuing the H2D copies (non_blocking, so this is mostly launch overhead)
    sync   torch.cuda.synchronize(): the actual DMA time lands here
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List, Optional

import torch
from torch import nn

from .store import AdapterStore, live_to_saved_key


class WeightJumper:
    def __init__(self, model: nn.Module, store: AdapterStore, adapter_name: str = "default",
                 key_map: Optional[Callable[[str], Optional[str]]] = None):
        self.model, self.store, self.adapter_name = model, store, adapter_name
        key_map = key_map or (lambda n: live_to_saved_key(n, adapter_name))
        self.live: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            k = key_map(n)
            if k is not None:
                self.live[k] = p.data
        if not self.live:
            raise ValueError("no live LoRA parameters found; is a PEFT adapter attached?")
        store.check_same_schema()
        any_name = next(iter(store.names()))
        store_keys, live_keys = set(store.keys(any_name)), set(self.live)
        if store_keys != live_keys:
            raise ValueError(f"store/live key mismatch: only-in-store {sorted(store_keys - live_keys)[:3]}, "
                             f"only-in-live {sorted(live_keys - store_keys)[:3]}")
        for k, src in store.get(any_name).items():
            if tuple(src.shape) != tuple(self.live[k].shape):
                raise ValueError(f"shape mismatch at {k}: store {tuple(src.shape)} live {tuple(self.live[k].shape)}")
        self.current: Optional[str] = None
        self.events: List[Dict[str, float]] = []
        self.nbytes = sum(t.numel() * t.element_size() for t in store.get(any_name).values())
        self.device = next(iter(self.live.values())).device

    @torch.no_grad()
    def jump(self, name: str, record: bool = True) -> Dict[str, float]:
        cuda = self.device.type == "cuda"
        mem0 = torch.cuda.memory_allocated() if cuda else 0
        t0 = time.perf_counter()
        src = self.store.get(name)
        t1 = time.perf_counter()
        for k, dst in self.live.items():
            dst.copy_(src[k], non_blocking=True)
        t2 = time.perf_counter()
        if cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        ev = {"src": self.current, "dst": name,
              "t_fetch_ms": (t1 - t0) * 1e3, "t_copy_ms": (t2 - t1) * 1e3, "t_sync_ms": (t3 - t2) * 1e3,
              "t_total_ms": (t3 - t0) * 1e3, "mb": self.nbytes / 1e6,
              "cuda_mem_delta": (torch.cuda.memory_allocated() - mem0) if cuda else 0}
        self.current = name
        if record:
            self.events.append(ev)
        return ev

    def benchmark(self, names: List[str], trials: int = 200, warmup: int = 5, monitor=None, tag: str = "") -> Dict[str, float]:
        for i in range(warmup):
            self.jump(names[i % len(names)], record=False)
        evs = []
        for i in range(trials):
            ev = self.jump(names[i % len(names)])
            evs.append(ev)
            if monitor:
                monitor.log("jump", tag=tag, **ev)
        tot = sorted(e["t_total_ms"] for e in evs)
        q = lambda xs, p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
        summ = {"tag": tag, "trials": trials, "mb_per_jump": self.nbytes / 1e6,
                "p50_ms": statistics.median(tot), "p95_ms": q(tot, 0.95), "p99_ms": q(tot, 0.99),
                "max_ms": tot[-1], "mean_ms": statistics.fmean(tot),
                "fetch_p50_ms": statistics.median(e["t_fetch_ms"] for e in evs),
                "copy_p50_ms": statistics.median(e["t_copy_ms"] for e in evs),
                "sync_p50_ms": statistics.median(e["t_sync_ms"] for e in evs),
                "gbps": (self.nbytes / 1e9) / (statistics.median(tot) / 1e3),
                "max_cuda_mem_delta": max(e["cuda_mem_delta"] for e in evs)}
        if monitor:
            monitor.log("jump_benchmark", **summ)
        return summ


# ----------------------------------------------------------------------------- correctness
@torch.no_grad()
def greedy_ids(model, tok, prompt: str, system: str, max_new_tokens: int = 48) -> List[int]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                  return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return out[0][enc["input_ids"].shape[1]:].tolist()


def equivalence_report(jumped_ids: List[int], reference_ids: List[int]) -> Dict[str, object]:
    n = min(len(jumped_ids), len(reference_ids))
    first_diff = next((i for i in range(n) if jumped_ids[i] != reference_ids[i]), None)
    identical = first_diff is None and len(jumped_ids) == len(reference_ids)
    return {"identical": identical, "first_diff_pos": first_diff, "len_jumped": len(jumped_ids), "len_ref": len(reference_ids)}
