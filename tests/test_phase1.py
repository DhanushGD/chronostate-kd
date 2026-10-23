"""Phase 1 tests that need no model download: a fake tokenizer stands in for Qwen's."""
import torch

from chronostate.data import IGNORE_INDEX, build_dataset, collate, dataset_stats, format_example
from chronostate.evaluate import extract_code, extract_gsm8k_gold, extract_last_number, run_python_with_tests
from chronostate.teacher import cache_teacher_logits, load_shard, shard_path, verify_shard


class FakeTok:
    """Deterministic char-level tokenizer: id = ord(c) % 97 + 3. eos=2, pad=0. Chat template = fixed prefix."""
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 97 + 3 for c in text]}

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True):
        text = "<sys><usr>" + msgs[-1]["content"] + "<asst>"   # system text omitted to keep prompts short
        return self(text)["input_ids"]


class Cfg:
    class data:
        max_len = 64
        rows_per_adapter = 10
        seed = 0


def test_format_masks_prompt_and_appends_eos():
    tok = FakeTok()
    ex = format_example(tok, "code", "hi", "print(1)", max_len=256)
    n = ex["n_prompt"]
    assert all(l == IGNORE_INDEX for l in ex["labels"][:n])
    assert ex["labels"][n:] == ex["input_ids"][n:]
    assert ex["input_ids"][-1] == tok.eos_token_id


def test_format_truncates_and_rejects_long_prompts():
    tok = FakeTok()
    ex = format_example(tok, "reasoning", "q", "x" * 500, max_len=64)
    assert len(ex["input_ids"]) == 64 and len(ex["labels"]) == 64
    assert format_example(tok, "reasoning", "q" * 100, "a", max_len=64) is None


def test_build_dataset_assigns_contiguous_row_ids_and_stats():
    tok = FakeTok()
    rows = [{"prompt": f"p{i}", "response": "ans " * (i + 1)} for i in range(10)]
    ds = build_dataset(Cfg, tok, "code", rows=rows)
    assert [e["row_id"] for e in ds] == list(range(len(ds)))
    st = dataset_stats(ds)
    assert st["n"] == len(ds) and st["total_response_tokens"] > 0


def test_collate_right_pads_and_keeps_alignment():
    b = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "row_id": 0},
         {"input_ids": [8, 9], "labels": [-100, 9], "row_id": 1}]
    enc = collate(b, pad_id=0)
    assert enc["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert enc["labels"].tolist() == [[-100, 6, 7], [-100, 9, -100]]
    assert enc["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert enc["row_id"].tolist() == [0, 1]


class OracleTeacher(torch.nn.Module):
    """Puts its top-1 logit on the *next* token of the input, so alignment checks must hit ~100%."""
    V = 120

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        logits = torch.randn(B, T, self.V) * 0.01
        nxt = input_ids[:, 1:]
        logits[:, :-1].scatter_(-1, nxt.unsqueeze(-1), 10.0)
        return type("O", (), {"logits": logits})()


def test_teacher_cache_roundtrip_resume_and_alignment(tmp_path):
    tok = FakeTok()
    rows = [{"prompt": f"p{i}", "response": "hello world " * (i % 4 + 1)} for i in range(12)]
    ds = build_dataset(Cfg, tok, "code", rows=rows)
    paths = cache_teacher_logits(OracleTeacher(), ds, tmp_path, "code", pad_id=0, max_len=64,
                                 top_k=8, batch_size=5, shard_rows=5, device="cpu")
    assert len(paths) == 3 and all(p.exists() for p in paths)
    sd = load_shard(paths[0])
    assert sd["values"].shape == (5, 64, 8) and sd["values"].dtype == torch.float16
    assert sd["indices"].dtype == torch.int32 and sd["row_ids"].tolist() == [0, 1, 2, 3, 4]
    assert sd["lengths"].tolist() == [len(e["input_ids"]) for e in ds[:5]]
    rep = verify_shard(paths[0], ds)
    assert rep["teacher_top1_agreement"] > 0.99
    # resume: deleting shard 1 and re-running recreates only it
    paths[1].unlink()
    mtime0 = paths[0].stat().st_mtime
    cache_teacher_logits(OracleTeacher(), ds, tmp_path, "code", pad_id=0, max_len=64,
                         top_k=8, batch_size=5, shard_rows=5, device="cpu")
    assert paths[1].exists() and paths[0].stat().st_mtime == mtime0
    assert shard_path(tmp_path, "code", 1) == paths[1]


def test_gsm8k_parsing():
    assert extract_gsm8k_gold("blah blah\n#### 1,234") == "1234"
    assert extract_last_number("so the answer is 7 apples, #### 7.0") == "7"
    assert extract_last_number("costs $3.50 in total") == "3.5"
    assert extract_last_number("no numbers here") is None


def test_code_extraction_and_execution():
    txt = "Sure:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    code = extract_code(txt)
    assert code.startswith("def add")
    assert run_python_with_tests(code, [], ["assert add(1, 2) == 3"])["passed"]
    assert not run_python_with_tests(code, [], ["assert add(1, 2) == 4"])["passed"]
    assert run_python_with_tests("while True: pass", [], ["assert True"], timeout=1)["error"] == "timeout"
