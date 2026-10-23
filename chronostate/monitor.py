"""Event logging for every phase (plan §6).

One JSON object per line in results/events.jsonl. Every number in the final results table
is computed from these lines, never typed by hand. Weights & Biases is optional: when
`wandb` is importable and WANDB_API_KEY is set, `log()` also forwards numeric fields.

Usage:
    mon = Monitor("results/events.jsonl", run="phase2-adapter-code")
    mon.log("train_step", step=10, loss=1.23, ce=1.0, kl=0.3)
    mon.log("jump", src="code", dst="reasoning", t_total_ms=12.4)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:  # optional
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


def gpu_mem_gb() -> Dict[str, float]:
    """Current/peak CUDA memory in GB, or zeros on CPU. Safe to call anywhere."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "vram_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
    except Exception:
        pass
    return {"vram_alloc_gb": 0.0, "vram_peak_gb": 0.0}


class Monitor:
    def __init__(self, path: str | os.PathLike, run: str, use_wandb: bool = False, config: Optional[dict] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = run
        self._fh = self.path.open("a", encoding="utf-8")
        self._wb = None
        if use_wandb and wandb is not None and os.environ.get("WANDB_API_KEY"):
            self._wb = wandb.init(project="chronostate-kd", name=run, config=config or {}, reinit=True)

    def log(self, event: str, **fields: Any) -> Dict[str, Any]:
        rec = {"ts": time.time(), "run": self.run, "event": event, **fields}
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        if self._wb is not None:
            numeric = {k: v for k, v in fields.items() if isinstance(v, (int, float)) and k != "step"}
            self._wb.log(numeric, step=fields.get("step"))
        return rec

    def close(self) -> None:
        self._fh.close()
        if self._wb is not None:
            self._wb.finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_events(path: str | os.PathLike, event: Optional[str] = None, run: Optional[str] = None):
    """Load events.jsonl back as a list of dicts, optionally filtered."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if event and rec.get("event") != event:
                continue
            if run and rec.get("run") != run:
                continue
            out.append(rec)
    return out
