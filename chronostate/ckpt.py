"""Resumable checkpoints for Colab sessions that die.

Every long-running phase writes to a Drive-backed directory and keeps a manifest.json
listing what has been saved. On restart, `latest()` tells the phase where to resume.

Why a manifest rather than listing the directory: Drive listing of many files is slow
and occasionally stale right after a write; a single small JSON is reliable.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/chronostate"


def drive_root() -> Path:
    """Drive root on Colab, or ./drive_local when running elsewhere (tests, local dev)."""
    root = Path(os.environ.get("CHRONOSTATE_ROOT", DEFAULT_DRIVE_ROOT))
    if not root.parent.exists():  # /content/drive not mounted -> local fallback
        root = Path("drive_local")
    root.mkdir(parents=True, exist_ok=True)
    return root


class CheckpointManager:
    def __init__(self, directory: str | os.PathLike, every_steps: int = 50, keep_last: int = 2):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every_steps = every_steps
        self.keep_last = keep_last
        self.manifest_path = self.dir / "manifest.json"
        self.manifest: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"entries": []}

    def _save(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2))
        tmp.replace(self.manifest_path)

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.every_steps == 0

    def record(self, step: int, path: str | os.PathLike, **extra: Any) -> Dict[str, Any]:
        """Register a checkpoint that the caller has already written to `path`."""
        entry = {"step": step, "path": str(path), "wall": time.time(), **extra}
        self.manifest["entries"].append(entry)
        self.manifest["entries"].sort(key=lambda e: e["step"])
        self._prune()
        self._save()
        return entry

    def _prune(self) -> None:
        entries: List[Dict[str, Any]] = self.manifest["entries"]
        while len(entries) > self.keep_last:
            old = entries.pop(0)
            p = Path(old["path"])
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink(missing_ok=True)

    def latest(self) -> Optional[Dict[str, Any]]:
        entries = self.manifest["entries"]
        return entries[-1] if entries else None
