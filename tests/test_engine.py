"""Engine bookkeeping tests with a fake tokenizer and a fake HF-like model (no downloads).
They check what Colab cannot cheaply: the id/cache prefix invariant across turns and modes,
jump dispatch through the router, and the per-turn record."""
import torch
from transformers import DynamicCache

from chronostate.engine import ChronoStateEngine, session_summary, transcript_markdown
from chronostate.router import HybridRouter, build_prototypes
from tests.test_router import POOLS, fake_embed

IM_START, IM_END, PAD = 1, 2, 0


class FakeTok:
    pad_token_id = PAD

    def convert_tokens_to_ids(self, t):
        return {"<|im_start|>": IM_START, "<|im_end|>": IM_END}[t]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 90 + 10 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i >= 10)


class FakeModel:
    """generate(): prefills every id not yet in the cache (one cache entry per token), then emits
    `reply` ids followed by <|im_end|>, honouring max_new_tokens. Records prefill sizes."""

    def __init__(self, reply=(65, 66, 67)):
        self.reply = list(reply)
        self.prefills = []

    def generate(self, input_ids, attention_mask, past_key_values, max_new_tokens, **kw):
        cache = past_key_values
        n_in = input_ids.shape[1]
        have = cache.get_seq_length()
        assert have <= n_in, "cache longer than input: prefix invariant broken"
        self.prefills.append(n_in - have)
        seq = input_ids[0].tolist()
        # how many reply tokens are already emitted since the last im_start+assistant header?
        emitted = 0
        for t in reversed(seq):
            if t in self.reply:
                emitted += 1
            else:
                break
        new = []
        for _ in range(max_new_tokens):
            if emitted < len(self.reply):
                new.append(self.reply[emitted]); emitted += 1
            else:
                new.append(IM_END); break
        total = n_in + len(new)
        k = torch.zeros(1, 1, total - 1 - have, 1)              # cache covers all but the last token
        cache.update(k, k, 0)
        out = type("O", (), {})()
        out.sequences = torch.tensor([seq + new])
        out.past_key_values = cache
        return out


def _engine(kv_mode, model=None):
    protos = build_prototypes(fake_embed, POOLS)
    router = HybridRouter(fake_embed, protos, beta=0.7)
    jumps = []
    jumper = type("J", (), {"jump": lambda self, n: (jumps.append(n) or {"t_total_ms": 1.5})})()
    eng = ChronoStateEngine(model or FakeModel(), FakeTok(), router, jumper, kv_mode=kv_mode, max_new_tokens=8, device="cpu")
    return eng, jumps


def test_reuse_mode_prefills_only_new_tokens_and_keeps_prefix():
    eng, jumps = _engine("reuse")
    turns = eng.run_session(["write a python function", "why?", "how many apples in total", "and the total cost in dollars"])
    assert [t.chosen for t in turns] == ["code", "code", "reasoning", "reasoning"]
    assert [t.jumped for t in turns] == [True, False, True, False]
    assert jumps == ["code", "reasoning"] and turns[2].jump_ms == 1.5 and turns[1].jump_ms == 0.0
    m = eng.model
    # first turn prefills system + user; later turns prefill only the new user turn (+ the closed assistant tail)
    assert m.prefills[0] > turns[0].prompt_tokens_new             # system prompt + first user turn
    assert m.prefills[2] == turns[1].prompt_tokens_new + 1 + len(eng.nl)   # new user turn + last generated id (never fed) + closing newline
    assert all(p < m.prefills[0] for p in m.prefills[2::2])          # every later first-token prefill is small
    assert turns[1].cache_tokens_before > 0 and turns[1].cache_tokens_before < len(eng.ids)
    assert eng.cache.get_seq_length() == len(eng.ids) - 1 - len(eng.nl) or eng.cache.get_seq_length() <= len(eng.ids)
    assert all(t.answer == "ABC" for t in turns) and all(t.gen_tokens == 4 for t in turns)


def test_recompute_mode_prefills_everything_each_turn():
    eng, _ = _engine("recompute")
    turns = eng.run_session(["write a python function", "why?", "how many apples in total"])
    assert all(t.cache_tokens_before == 0 for t in turns)
    first_token_prefills = eng.model.prefills[0::2]
    assert first_token_prefills == sorted(first_token_prefills) and first_token_prefills[-1] > first_token_prefills[0]


def test_summary_and_transcript():
    eng, _ = _engine("reuse")
    turns = eng.run_session(["write a python function", "how many apples in total"])
    s = session_summary(turns)
    assert s["turns"] == 2 and s["jumps"] == 1 and s["gen_tokens_total"] == 8
    md = transcript_markdown(turns, "t")
    assert "JUMP → reasoning" in md and "**User:** why?" not in md and md.count("### Turn") == 2
