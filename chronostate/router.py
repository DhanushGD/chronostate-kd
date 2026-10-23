"""Hybrid query router (Phase 4, spec §4.2).

    score_j = beta * cos(E_query, E_j) + (1 - beta) * T[prev, j]

E_j is the mean-pooled, L2-normalised embedding of ~200 prompts of domain j (a prototype),
T is the Markov transition matrix from MarkovStateLayer. A weight jump is dispatched when
argmax_j score_j != previous state. beta = 1 is a pure semantic classifier (the ablation
baseline); beta = 0.7 is the default.

Cosine similarities between a prompt and two prototypes are both typically in 0.3–0.6 and
differ by a few hundredths, while the prior row differs by 0.6 (0.8 vs 0.2). To keep
`beta` meaningful the cosine vector is turned into a distribution with
softmax(cos / cos_temperature) before mixing, so both terms are probabilities over the K
domains. A near-tie (a follow-up like "why?") maps to ~uniform and lets the prior decide;
clear evidence (gap ≥ 0.15 at temperature 0.05) maps to ~one-hot and overrides it.
(A first version used min-max normalisation; with K = 2 that always yields {0, 1} and
silently disables the prior. Caught by tests/test_router.py.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .markov import MarkovStateLayer

EmbedFn = Callable[[List[str]], torch.Tensor]   # list[str] -> (N, d) float tensor


def build_prototypes(embed: EmbedFn, prompts_by_domain: Dict[str, Sequence[str]], batch_size: int = 64) -> Dict[str, torch.Tensor]:
    protos = {}
    for domain, prompts in prompts_by_domain.items():
        chunks = [embed(list(prompts[i : i + batch_size])) for i in range(0, len(prompts), batch_size)]
        e = torch.cat(chunks).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        protos[domain] = torch.nn.functional.normalize(e, dim=0)
    return protos


@dataclass
class RouteDecision:
    turn: int
    query: str
    cos: Dict[str, float]
    cos_norm: Dict[str, float]
    prior: Dict[str, float]
    combined: Dict[str, float]
    prev: Optional[str]
    chosen: str
    jumped: bool
    beta: float
    latency_ms: float
    extra: Dict[str, float] = field(default_factory=dict)


class HybridRouter:
    def __init__(self, embed: EmbedFn, prototypes: Dict[str, torch.Tensor], beta: float = 0.7,
                 stickiness: float = 0.8, ema: float = 0.1, cos_temperature: float = 0.05,
                 on_jump: Optional[Callable[[str], None]] = None):
        self.embed = embed
        self.domains = list(prototypes)
        self.P = torch.stack([prototypes[d] for d in self.domains])          # (K, d)
        self.beta = float(beta)
        self.cos_temperature = float(cos_temperature)
        self.markov = MarkovStateLayer(self.domains, stickiness=stickiness, ema=ema)
        self.on_jump = on_jump
        self.turn = 0
        self.history: List[RouteDecision] = []

    @property
    def state(self) -> Optional[str]:
        return None if self.markov.prev_state is None else self.domains[self.markov.prev_state]

    def reset(self) -> None:
        """New conversation: forget the state, keep the learned T."""
        self.markov.reset()
        self.turn = 0

    @torch.no_grad()
    def score(self, query: str) -> Dict[str, torch.Tensor]:
        e = torch.nn.functional.normalize(self.embed([query]).float()[0], dim=0)
        cos = self.P @ e                                                      # (K,)
        cos_n = torch.softmax(cos / self.cos_temperature, dim=0)            # semantic 'probability' per domain
        prior = self.markov.prior()
        combined = self.beta * cos_n + (1.0 - self.beta) * prior
        return {"cos": cos, "cos_norm": cos_n, "prior": prior, "combined": combined}

    @torch.no_grad()
    def route(self, query: str, update: bool = True) -> RouteDecision:
        t0 = time.perf_counter()
        s = self.score(query)
        j = int(torch.argmax(s["combined"]))
        prev = self.state
        chosen = self.domains[j]
        jumped = prev is not None and chosen != prev
        if prev is None:
            jumped = True                                                     # first turn always loads an adapter
        if update:
            self.markov.update(j)
        if jumped and self.on_jump is not None:
            self.on_jump(chosen)
        d = RouteDecision(turn=self.turn, query=query,
                          cos={k: float(v) for k, v in zip(self.domains, s["cos"])},
                          cos_norm={k: float(v) for k, v in zip(self.domains, s["cos_norm"])},
                          prior={k: float(v) for k, v in zip(self.domains, s["prior"])},
                          combined={k: float(v) for k, v in zip(self.domains, s["combined"])},
                          prev=prev, chosen=chosen, jumped=jumped, beta=self.beta,
                          latency_ms=(time.perf_counter() - t0) * 1e3)
        self.history.append(d)
        self.turn += 1
        return d

    def transition_matrix(self) -> Dict[str, Dict[str, float]]:
        T = self.markov.T
        return {a: {b: float(T[i, j]) for j, b in enumerate(self.domains)} for i, a in enumerate(self.domains)}


def make_st_embedder(model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu") -> EmbedFn:
    """sentence-transformers embedder; 33M params, fine on CPU (~5 ms per short prompt)."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name, device=device)

    def embed(texts: List[str]) -> torch.Tensor:
        return torch.as_tensor(st.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64))

    return embed
