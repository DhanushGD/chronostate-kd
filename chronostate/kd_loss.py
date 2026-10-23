"""Knowledge-distillation loss over cached top-k teacher logits (spec §4.1).

    L = (1 - alpha) * CE(labels, student)  +  alpha * tau^2 * KL( softmax(T/tau) || softmax(S/tau) )

Why top-k: the full-vocab teacher distribution for 3k samples x 256 tokens x 151k vocab
would be ~750 GB. Keeping the teacher's top-64 (values + indices) per token is ~1 GB and
captures almost all of the probability mass. The student's logits are *gathered at the
teacher's indices* (never at the student's own top-k), then both are renormalised over
those k entries before the KL.

Alignment convention: logits at position t predict token t+1. Everything passed in
(student_logits, labels, teacher values/indices) is indexed by *position*, so this
function performs the shift itself when `shift=True` (the default). The teacher cache
stores logits at the same positions, so no separate shift is needed there.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class KDLossOutput:
    loss: torch.Tensor
    ce: torch.Tensor
    kl: torch.Tensor
    n_tokens: int


def kd_loss(
    student_logits: torch.Tensor,        # (B, T, V) any float dtype
    labels: torch.Tensor,                # (B, T)   long, IGNORE_INDEX on prompt/pad tokens
    teacher_topk_values: torch.Tensor,   # (B, T, K) teacher logits (raw, un-softmaxed)
    teacher_topk_indices: torch.Tensor,  # (B, T, K) long/int32 vocab ids
    alpha: float = 0.7,
    tau: float = 2.0,
    shift: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> KDLossOutput:
    if shift:
        student_logits = student_logits[:, :-1]
        labels = labels[:, 1:]
        teacher_topk_values = teacher_topk_values[:, :-1]
        teacher_topk_indices = teacher_topk_indices[:, :-1]

    V = student_logits.size(-1)
    flat_logits = student_logits.reshape(-1, V).float()
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)

    mask = labels.ne(ignore_index)
    n_tokens = int(mask.sum().item())

    if alpha == 0.0 or n_tokens == 0:
        zero = torch.zeros((), device=student_logits.device, dtype=ce.dtype)
        return KDLossOutput(loss=ce, ce=ce.detach(), kl=zero, n_tokens=n_tokens)

    idx = teacher_topk_indices.long()
    s_topk = torch.gather(student_logits, dim=-1, index=idx).float()   # (B, T, K)
    t_topk = teacher_topk_values.float()

    log_p_s = F.log_softmax(s_topk / tau, dim=-1)
    p_t = F.softmax(t_topk / tau, dim=-1)
    kl_tok = F.kl_div(log_p_s, p_t, reduction="none").sum(-1)         # (B, T)
    kl = (kl_tok * mask).sum() / n_tokens * (tau ** 2)

    loss = (1.0 - alpha) * ce + alpha * kl
    return KDLossOutput(loss=loss, ce=ce.detach(), kl=kl.detach(), n_tokens=n_tokens)
