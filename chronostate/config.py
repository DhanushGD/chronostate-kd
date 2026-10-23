"""Load configs/*.yaml into a plain nested namespace. Every knob lives in YAML so the SFT
ablation (lite_sft.yaml) and the scale-up (full.yaml) are config diffs, not code edits."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _ns(raw)
    cfg._path = str(path)
    return cfg


def to_dict(cfg: Any) -> Any:
    if isinstance(cfg, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(cfg).items() if not k.startswith("_")}
    if isinstance(cfg, list):
        return [to_dict(v) for v in cfg]
    return cfg
