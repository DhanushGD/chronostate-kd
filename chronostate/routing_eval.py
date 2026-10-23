"""Synthetic multi-turn routing benchmark (Phase 4).

Conversations are built from held-out prompt pools (never the prototype prompts) as runs
of 2–4 turns in one domain followed by a switch. Inside a run, some turns are *follow-ups*:
short, domain-agnostic messages ("why?", "and if it were 5 instead?") whose ground-truth
domain is the one the conversation is already in. A pure semantic classifier has no signal
on those; the Markov prior does. That is the behaviour the beta sweep isolates.

Metrics per beta:
    accuracy            chosen == label over all turns
    followup_accuracy   same, over follow-up turns only
    jumps               number of adapter swaps dispatched (first turn excluded)
    unnecessary_jumps   swaps on turns whose label did not change   (cost: latency + wrong adapter)
    missed_switches     turns whose label changed but no swap       (cost: wrong adapter)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .router import HybridRouter

FOLLOWUPS = [
    "why?", "can you explain that again?", "and if it were 5 instead?", "what about the edge cases?",
    "shorter please", "go on", "is that the final answer?", "double-check that", "ok, and the next step?",
    "explain line by line", "can you simplify it?", "what if the input is empty?", "redo it more carefully",
    "hmm, are you sure?", "give me one more example", "and in the general case?", "thanks, continue",
    "what does that give?", "same thing but with 12", "walk me through it slowly",
]


@dataclass
class Turn:
    text: str
    label: str
    is_followup: bool
    label_changed: bool


def make_conversations(pools: Dict[str, Sequence[str]], n_conv: int = 40, turns: int = 6,
                       followup_frac: float = 0.3, seed: int = 0) -> List[List[Turn]]:
    rng = random.Random(seed)
    domains = list(pools)
    convs: List[List[Turn]] = []
    for _ in range(n_conv):
        conv: List[Turn] = []
        domain = rng.choice(domains)
        run_left = rng.randint(2, 4)
        prev = None
        while len(conv) < turns:
            if run_left == 0:
                domain = rng.choice([d for d in domains if d != domain])
                run_left = rng.randint(2, 4)
            # a follow-up never starts a run (the first turn of a topic must carry the topic)
            is_fu = prev == domain and rng.random() < followup_frac
            text = rng.choice(FOLLOWUPS) if is_fu else rng.choice(pools[domain])
            conv.append(Turn(text=text, label=domain, is_followup=is_fu, label_changed=(prev is not None and prev != domain)))
            prev = domain
            run_left -= 1
        convs.append(conv)
    return convs


def run_routing_eval(router: HybridRouter, convs: List[List[Turn]], learn: bool = True) -> Dict[str, float]:
    n = correct = fu_n = fu_correct = jumps = unnecessary = missed = 0
    lat = []
    for conv in convs:
        router.reset()
        for i, t in enumerate(conv):
            d = router.route(t.text, update=learn)
            n += 1; correct += d.chosen == t.label; lat.append(d.latency_ms)
            if t.is_followup:
                fu_n += 1; fu_correct += d.chosen == t.label
            if i > 0:
                jumps += d.jumped
                unnecessary += d.jumped and not t.label_changed
                missed += (not d.jumped) and t.label_changed
    switches = sum(t.label_changed for c in convs for t in c)
    return {"beta": router.beta, "turns": n, "accuracy": correct / n,
            "followup_turns": fu_n, "followup_accuracy": fu_correct / max(1, fu_n),
            "true_switches": switches, "jumps": jumps, "unnecessary_jumps": unnecessary, "missed_switches": missed,
            "jumps_per_conv": jumps / len(convs), "latency_p50_ms": sorted(lat)[len(lat) // 2]}
