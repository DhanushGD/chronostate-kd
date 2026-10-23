"""Markov state layer (spec §4.2): P(S_t = j | S_{t-1} = i) = T[i, j].

Holds a K x K row-stochastic transition matrix T, the previous state, and an EMA update.
The router combines T[prev, :] with a semantic score; see router.py (Phase 4).

Why EMA rather than raw counts: with ~40 eval conversations the count matrix is noisy and
a single long conversation would dominate. EMA with a small rate (0.1) keeps T close to
its "sticky" prior and adapts slowly, which is the behaviour we want to demonstrate
(fewer unnecessary jumps), and it keeps every row stochastic by construction.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


class MarkovStateLayer(nn.Module):
    def __init__(
        self,
        state_names: Sequence[str],
        stickiness: float = 0.8,
        ema: float = 0.1,
        init_T: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.state_names = list(state_names)
        K = len(self.state_names)
        if K < 2:
            raise ValueError("need at least two states")
        self.ema = float(ema)

        if init_T is None:
            off = (1.0 - stickiness) / (K - 1)
            T = torch.full((K, K), off, dtype=torch.float32)
            T.fill_diagonal_(stickiness)
        else:
            T = init_T.clone().float()
            if T.shape != (K, K):
                raise ValueError(f"init_T must be {K}x{K}")
            T = T / T.sum(-1, keepdim=True)
        self.register_buffer("T", T)
        self.prev_state: Optional[int] = None
        self.n_updates = 0

    @property
    def K(self) -> int:
        return len(self.state_names)

    def index(self, name: str) -> int:
        return self.state_names.index(name)

    def reset(self) -> None:
        """Start a new conversation: forget the previous state, keep the learned T."""
        self.prev_state = None

    def prior(self) -> torch.Tensor:
        """Routing prior over next states. Uniform when no previous state exists."""
        if self.prev_state is None:
            return torch.full((self.K,), 1.0 / self.K, dtype=self.T.dtype, device=self.T.device)
        return self.T[self.prev_state]

    @torch.no_grad()
    def update(self, new_state: int) -> None:
        """Observe the chosen state. Moves T[prev] toward one-hot(new_state) by `ema`."""
        if not 0 <= new_state < self.K:
            raise IndexError(new_state)
        if self.prev_state is not None:
            row = self.T[self.prev_state]
            onehot = torch.zeros_like(row)
            onehot[new_state] = 1.0
            self.T[self.prev_state] = (1.0 - self.ema) * row + self.ema * onehot
            self.n_updates += 1
        self.prev_state = new_state

    def is_row_stochastic(self, atol: float = 1e-5) -> bool:
        return bool(torch.allclose(self.T.sum(-1), torch.ones(self.K), atol=atol)) and bool((self.T >= 0).all())

    def extra_repr(self) -> str:
        return f"states={self.state_names}, ema={self.ema}, prev={self.prev_state}"
