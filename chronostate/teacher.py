"""Teacher top-k logit cache (Phase 1).

For every row of the saved dataset, run the teacher once and keep, at every position t,
the top-k logit values (fp16) and their vocab indices (int32). Rows are written in shards
of `shard_rows` to <cache_dir>/<domain>/shard-NNNN.safetensors. A shard that already
exists is skipped, so a dead Colab session resumes at the shard boundary.

Shard layout (all tensors padded to max_len on the right, matching data.collate):
    values   (N, max_len, K)  float16   raw teacher logits (NOT softmaxed; kd_loss applies tau)
    indices  (N, max_len, K)  int32     vocab ids (151,665 > int16, so int32)
    lengths  (N,)             int32     true sequence length of each row
    row_ids  (N,)             int32     dataset row ids, for an alignment assert in Phase 2

Why full positions rather than response-only: simpler indexing, and kd_loss already masks
prompt positions via labels. Cost for 3k rows x 256 x 64: ~150 MB values + ~300 MB indices.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file

from .data import IGNORE_INDEX, collate


def shard_path(cache_dir, domain: str, shard_idx: int) -> Path:
    return Path(cache_dir) / domain / f"shard-{shard_idx:04d}.safetensors"


def n_shards(n_rows: int, shard_rows: int) -> int:
    return math.ceil(n_rows / shard_rows)


@torch.no_grad()
def cache_teacher_logits(
    model,
    examples: List[Dict[str, Any]],
    cache_dir,
    domain: str,
    pad_id: int,
    max_len: int,
    top_k: int = 64,
    batch_size: int = 8,
    shard_rows: int = 500,
    monitor=None,
    device: str = "cuda",
) -> List[Path]:
    out_dir = Path(cache_dir) / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    paths = []
    total_shards = n_shards(len(examples), shard_rows)

    for s in range(total_shards):
        p = shard_path(cache_dir, domain, s)
        paths.append(p)
        if p.exists():
            print(f"[teacher:{domain}] shard {s}/{total_shards} exists, skipping")
            continue
        rows = examples[s * shard_rows : (s + 1) * shard_rows]
        N = len(rows)
        values = torch.zeros((N, max_len, top_k), dtype=torch.float16)
        indices = torch.zeros((N, max_len, top_k), dtype=torch.int32)
        lengths = torch.zeros((N,), dtype=torch.int32)
        row_ids = torch.tensor([r["row_id"] for r in rows], dtype=torch.int32)

        t0 = time.time()
        for b in range(0, N, batch_size):
            batch = rows[b : b + batch_size]
            enc = collate(batch, pad_id=pad_id, max_len=max_len)
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits                                            # (B, T, V) fp16
            tv, ti = logits.float().topk(top_k, dim=-1)         # topk in fp32 for stable ordering
            values[b : b + len(batch)] = tv.to(torch.float16).cpu()
            indices[b : b + len(batch)] = ti.to(torch.int32).cpu()
            lengths[b : b + len(batch)] = torch.tensor([len(x["input_ids"]) for x in batch], dtype=torch.int32)
            del logits, tv, ti

        tmp = p.with_suffix(".tmp.safetensors")
        save_file({"values": values, "indices": indices, "lengths": lengths, "row_ids": row_ids}, str(tmp))
        tmp.replace(p)
        dt = time.time() - t0
        print(f"[teacher:{domain}] shard {s + 1}/{total_shards}: {N} rows in {dt:.0f}s "
              f"({N * max_len / dt:.0f} tok/s), {p.stat().st_size / 1e6:.0f} MB")
        if monitor:
            monitor.log("teacher_shard", domain=domain, shard=s, rows=N, seconds=dt,
                        mb=p.stat().st_size / 1e6,
                        vram_peak_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0)
    return paths


def load_shard(path) -> Dict[str, torch.Tensor]:
    return load_file(str(path))


def verify_shard(path, examples: List[Dict[str, Any]], sample: int = 64) -> Dict[str, float]:
    """Alignment sanity check. Logits at position t predict token t+1, so the teacher's top-1
    at t should frequently equal labels[t+1] on response tokens. A strong teacher on its own
    training-style data gets well above 50%; near 1/V means the shard is misaligned."""
    sd = load_shard(path)
    hits = total = 0
    for i in range(min(sample, sd["values"].shape[0])):
        rid = int(sd["row_ids"][i])
        ex = examples[rid]
        L = int(sd["lengths"][i])
        assert L == len(ex["input_ids"]), f"length mismatch at row {rid}"
        top1 = sd["indices"][i, : L - 1, 0].long()                      # positions 0..L-2
        nxt = torch.tensor(ex["labels"][1:L])                           # tokens 1..L-1
        mask = nxt.ne(IGNORE_INDEX)
        hits += int((top1[mask] == nxt[mask]).sum())
        total += int(mask.sum())
    agreement = hits / max(1, total)
    return {"rows_checked": min(sample, sd["values"].shape[0]), "response_tokens": total, "teacher_top1_agreement": agreement}


def cache_summary(cache_dir, domain: str) -> Dict[str, Any]:
    d = Path(cache_dir) / domain
    files = sorted(d.glob("shard-*.safetensors"))
    rows = 0
    for f in files:
        rows += int(load_shard(f)["lengths"].shape[0])
    return {"domain": domain, "shards": len(files), "rows": rows, "mb": sum(f.stat().st_size for f in files) / 1e6}
