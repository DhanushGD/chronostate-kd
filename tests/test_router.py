"""Phase 4 tests with a keyword 'embedder' standing in for bge-small."""
import torch

from chronostate.router import HybridRouter, build_prototypes
from chronostate.routing_eval import FOLLOWUPS, make_conversations, run_routing_eval

CODE_WORDS = {"python", "function", "list", "string", "class", "loop"}
MATH_WORDS = {"apples", "dollars", "speed", "total", "km", "cost"}


def fake_embed(texts):
    """3-d vector: (code evidence, math evidence, neutral). Follow-ups land on the neutral axis."""
    out = []
    for t in texts:
        w = set(t.lower().replace("?", "").replace(",", "").split())
        c, m = len(w & CODE_WORDS), len(w & MATH_WORDS)
        out.append([c, m, 1.0 if (c == 0 and m == 0) else 0.1])
    return torch.tensor(out, dtype=torch.float32)


POOLS = {"code": ["write a python function", "reverse a string in python", "sort a list", "define a class with a loop"],
         "reasoning": ["how many apples in total", "speed in km per hour", "total cost in dollars", "cost of 3 apples"]}


def _router(beta, **kw):
    protos = build_prototypes(fake_embed, POOLS)
    return HybridRouter(fake_embed, protos, beta=beta, **kw)


def test_prototypes_are_unit_norm_and_separated():
    protos = build_prototypes(fake_embed, POOLS)
    for p in protos.values():
        assert abs(float(p.norm()) - 1.0) < 1e-5
    assert float(protos["code"] @ protos["reasoning"]) < 0.5


def test_beta_one_is_pure_semantic_and_first_turn_jumps():
    r = _router(1.0)
    d = r.route("write a python function")
    assert d.chosen == "code" and d.jumped and d.prev is None
    assert d.combined == d.cos_norm                      # prior ignored at beta=1
    d2 = r.route("how many apples in total")
    assert d2.chosen == "reasoning" and d2.jumped and d2.prev == "code"


def test_prior_resolves_ambiguous_followups_but_not_at_beta_one():
    r1 = _router(1.0); r7 = _router(0.7)
    for r in (r1, r7):
        r.route("total cost in dollars")
    # a follow-up has (almost) no semantic signal: softmax over cosines is near uniform
    d1 = r1.route("why?"); d7 = r7.route("why?")
    assert abs(d1.cos_norm["code"] - d1.cos_norm["reasoning"]) < 0.2
    assert d7.chosen == "reasoning" and not d7.jumped          # sticky prior wins at beta=0.7
    assert d1.chosen == "code" and d1.jumped                   # beta=1 follows the tiny tie-break: an unnecessary jump


def test_strong_semantic_evidence_still_overrides_prior():
    r = _router(0.7)
    r.route("total cost in dollars")
    d = r.route("write a python function with a loop")
    assert d.chosen == "code" and d.jumped


def test_markov_T_learns_from_routing_and_reset_keeps_it():
    r = _router(0.7)
    before = r.markov.T.clone()
    for q in ["write a python function", "sort a list", "reverse a string in python"]:
        r.route(q)
    assert r.markov.T[0, 0] > before[0, 0]                     # code->code reinforced
    r.reset()
    assert r.state is None and torch.equal(r.markov.T, r.markov.T)
    assert r.markov.is_row_stochastic()


def test_conversation_generator_shapes_and_labels():
    convs = make_conversations(POOLS, n_conv=20, turns=6, followup_frac=0.5, seed=1)
    assert len(convs) == 20 and all(len(c) == 6 for c in convs)
    for c in convs:
        assert not c[0].is_followup and not c[0].label_changed
        for a, b in zip(c, c[1:]):
            assert b.label_changed == (a.label != b.label)
            if b.is_followup:
                assert b.text in FOLLOWUPS and a.label == b.label


def test_markov_prior_reduces_unnecessary_jumps_in_eval():
    convs = make_conversations(POOLS, n_conv=40, turns=6, followup_frac=0.4, seed=2)
    m1 = run_routing_eval(_router(1.0), convs)
    m7 = run_routing_eval(_router(0.7), convs)
    assert m7["unnecessary_jumps"] < m1["unnecessary_jumps"]
    assert m7["followup_accuracy"] > m1["followup_accuracy"]
    assert m7["accuracy"] >= m1["accuracy"]
    assert m7["missed_switches"] <= m1["missed_switches"] and m1["turns"] == 240   # the prior must not add missed switches
