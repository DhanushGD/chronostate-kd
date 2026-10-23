"""Dataset loading and formatting (Phase 1).

Produces, per domain, a list of examples {input_ids, labels, row_id} where labels mask the
prompt with -100 so both CE and KL are computed on response tokens only.

Why the formatted dataset is *saved to Drive* (data/<domain>.pt) and reused in Phase 2:
the teacher logit shards are indexed by row position. If Phase 2 re-tokenised with a
different seed, truncation or template, row i of the cache would no longer match row i of
the batch and the KL term would be garbage with no error. Saving the exact token ids once
makes the alignment a file identity, not a re-derivation.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch

IGNORE_INDEX = -100

SYSTEM_PROMPTS = {
    "code": "You are a helpful coding assistant. Answer with correct, concise Python code.",
    "reasoning": "You are a careful math tutor. Think step by step, then give the final answer after '####'.",
}


def _extract_code_rows(ds, filter_language: Optional[str], n: int, seed: int) -> List[Dict[str, str]]:
    # m-a-p/CodeFeedback-Filtered-Instruction: columns query, answer, resource, lang
    rows = []
    for ex in ds:
        if filter_language and str(ex.get("lang", "")).lower() != filter_language.lower():
            continue
        q, a = ex.get("query"), ex.get("answer")
        if q and a and len(a) < 4000:
            rows.append({"prompt": q.strip(), "response": a.strip()})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _extract_gsm8k_rows(ds, n: int, seed: int) -> List[Dict[str, str]]:
    # openai/gsm8k: columns question, answer ("... #### 42")
    rows = [{"prompt": ex["question"].strip(), "response": ex["answer"].strip()} for ex in ds]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_domain_rows(cfg, domain: str) -> List[Dict[str, str]]:
    from datasets import load_dataset

    d = getattr(cfg.data.domains, domain)
    n, seed = cfg.data.rows_per_adapter, cfg.data.seed
    if domain == "code":
        ds = load_dataset(d.dataset, split=d.split)
        return _extract_code_rows(ds, getattr(d, "filter_language", None), n, seed)
    if domain == "reasoning":
        ds = load_dataset(d.dataset, getattr(d, "config", None), split=d.split)
        return _extract_gsm8k_rows(ds, n, seed)
    raise ValueError(domain)


def format_example(tok, domain: str, prompt: str, response: str, max_len: int) -> Optional[Dict[str, Any]]:
    """Tokenise one chat example. Returns None if the prompt alone leaves < 16 tokens for the response."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPTS[domain]}, {"role": "user", "content": prompt}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "keys"):          # transformers >= 5 may return a BatchEncoding
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = list(prompt_ids)
    resp_ids = list(tok(response, add_special_tokens=False)["input_ids"]) + [tok.eos_token_id]
    if len(prompt_ids) > max_len - 16:
        return None
    ids = (prompt_ids + resp_ids)[:max_len]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + resp_ids)[:max_len]
    return {"input_ids": ids, "labels": labels, "n_prompt": len(prompt_ids)}


def build_dataset(cfg, tok, domain: str, rows: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    rows = rows if rows is not None else load_domain_rows(cfg, domain)
    out, skipped = [], 0
    for r in rows:
        ex = format_example(tok, domain, r["prompt"], r["response"], cfg.data.max_len)
        if ex is None:
            skipped += 1
            continue
        ex["row_id"] = len(out)
        out.append(ex)
    if skipped:
        print(f"[data:{domain}] skipped {skipped} rows whose prompt exceeded max_len-16")
    return out


def dataset_stats(examples: List[Dict[str, Any]]) -> Dict[str, float]:
    lens = [len(e["input_ids"]) for e in examples]
    resp = [sum(1 for l in e["labels"] if l != IGNORE_INDEX) for e in examples]
    return {
        "n": len(examples),
        "mean_len": sum(lens) / max(1, len(lens)),
        "max_len": max(lens) if lens else 0,
        "mean_response_tokens": sum(resp) / max(1, len(resp)),
        "total_response_tokens": sum(resp),
        "truncated_frac": sum(1 for e in examples if e["labels"][-1] != IGNORE_INDEX and len(e["input_ids"]) == max(lens)) / max(1, len(lens)),
    }


def save_dataset(examples: List[Dict[str, Any]], path) -> None:
    torch.save(examples, path)


def load_saved_dataset(path) -> List[Dict[str, Any]]:
    return torch.load(path)


def collate(batch: List[Dict[str, Any]], pad_id: int, max_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Right-pad to the longest sequence in the batch (or max_len). Right padding keeps
    position t of every row aligned with position t of its teacher shard."""
    T = max_len or max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        attn[i, :n] = 1
    out = {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
    if "row_id" in batch[0]:
        out["row_id"] = torch.tensor([b["row_id"] for b in batch])
    return out
