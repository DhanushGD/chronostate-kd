"""Small, fixed evaluation sets (Phase 1 for the teacher, Phase 5 for base/SFT/KD students).

GSM8K: first `n` test problems, exact match on the final number after '####' (model output:
       the last number in the response).
MBPP:  first `n` problems of the sanitized split, pass@1: the model's code block plus the
       dataset's asserts run in a subprocess with a timeout.

Everything here takes a `generate(list_of_message_lists) -> list_of_strings` callable so
the same code evaluates a plain model, a PEFT model, or the hot-swapped student.
Per-sample records are written as JSONL so results can be re-aggregated or diffed later.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .data import SYSTEM_PROMPTS

Generate = Callable[[List[List[Dict[str, str]]]], List[str]]

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_gsm8k_gold(answer: str) -> str:
    return _norm_num(answer.split("####")[-1])


def extract_last_number(text: str) -> Optional[str]:
    if "####" in text:
        text = text.split("####")[-1]
    nums = _NUM.findall(text)
    return _norm_num(nums[-1]) if nums else None


def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if m:
        return max(m, key=len)
    return text


def run_python_with_tests(code: str, test_imports: List[str], tests: List[str], timeout: float = 10.0) -> Dict[str, object]:
    src = "\n".join(test_imports) + "\n" + code + "\n\n" + "\n".join(tests) + "\nprint('__PASS__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=timeout)
        passed = r.returncode == 0 and "__PASS__" in r.stdout
        err = "" if passed else (r.stderr.strip().splitlines() or [""])[-1][:200]
        return {"passed": passed, "error": err}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout"}
    finally:
        Path(path).unlink(missing_ok=True)


def _batched(items, bs):
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


def eval_gsm8k(generate: Generate, n: int = 100, batch_size: int = 8, out_jsonl=None, tag: str = "") -> Dict[str, float]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    msgs = [[{"role": "system", "content": SYSTEM_PROMPTS["reasoning"]},
             {"role": "user", "content": ex["question"]}] for ex in ds]
    golds = [extract_gsm8k_gold(ex["answer"]) for ex in ds]
    preds, t0 = [], time.time()
    for chunk in _batched(msgs, batch_size):
        preds.extend(generate(chunk))
    records, correct = [], 0
    for i, (p, g) in enumerate(zip(preds, golds)):
        pn = extract_last_number(p)
        ok = pn is not None and pn == g
        correct += ok
        records.append({"idx": i, "gold": g, "pred": pn, "correct": ok, "raw": p[-400:]})
    if out_jsonl:
        _write_jsonl(out_jsonl, records, tag)
    return {"task": "gsm8k", "n": n, "accuracy": correct / n, "seconds": time.time() - t0}


def eval_mbpp(generate: Generate, n: int = 50, batch_size: int = 8, out_jsonl=None, tag: str = "") -> Dict[str, float]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test").select(range(n))
    msgs = []
    for ex in ds:
        tests = "\n".join(ex["test_list"])
        user = (f"{ex['prompt']}\n\nYour code should pass these tests:\n```python\n{tests}\n```\n"
                "Reply with a single ```python code block containing only the function.")
        msgs.append([{"role": "system", "content": SYSTEM_PROMPTS["code"]}, {"role": "user", "content": user}])
    preds, t0 = [], time.time()
    for chunk in _batched(msgs, batch_size):
        preds.extend(generate(chunk))
    records, passed = [], 0
    for i, (ex, p) in enumerate(zip(ds, preds)):
        code = extract_code(p)
        res = run_python_with_tests(code, list(ex.get("test_imports") or []), list(ex["test_list"]))
        passed += res["passed"]
        records.append({"idx": i, "task_id": ex["task_id"], "passed": res["passed"], "error": res["error"], "code": code[:1500]})
    if out_jsonl:
        _write_jsonl(out_jsonl, records, tag)
    return {"task": "mbpp", "n": n, "pass_at_1": passed / n, "seconds": time.time() - t0}


def _write_jsonl(path, records, tag):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({"tag": tag, **r}) + "\n")


def make_hf_generate(model, tok, max_new_tokens: int = 320, device: str = "cuda") -> Generate:
    """Batched greedy generation with left padding (required so every row ends at the same position)."""
    import torch

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    @torch.no_grad()
    def generate(batch_msgs: List[List[Dict[str, str]]]) -> List[str]:
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in batch_msgs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)

    return generate
