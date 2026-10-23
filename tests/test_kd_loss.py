import torch

from chronostate.kd_loss import IGNORE_INDEX, kd_loss


def _fake_batch(B=2, T=8, V=50, K=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(B, T, V, generator=g)
    teacher_full = torch.randn(B, T, V, generator=g)
    labels = torch.randint(0, V, (B, T), generator=g)
    labels[:, :3] = IGNORE_INDEX                      # prompt tokens masked
    t_vals, t_idx = teacher_full.topk(K, dim=-1)
    return student, labels, t_vals, t_idx, teacher_full


def test_alpha_zero_is_plain_ce():
    s, y, tv, ti, _ = _fake_batch()
    out = kd_loss(s, y, tv, ti, alpha=0.0)
    assert torch.isclose(out.loss, out.ce)
    assert out.kl.item() == 0.0


def test_kl_is_zero_when_student_matches_teacher():
    s, y, tv, ti, teacher_full = _fake_batch()
    out = kd_loss(teacher_full, y, tv, ti, alpha=1.0, tau=2.0)
    assert out.kl.item() < 1e-5


def test_kl_positive_and_loss_is_convex_combo():
    s, y, tv, ti, _ = _fake_batch()
    a, tau = 0.7, 2.0
    out = kd_loss(s, y, tv, ti, alpha=a, tau=tau)
    assert out.kl.item() > 0
    expected = (1 - a) * out.ce + a * out.kl
    assert torch.isclose(out.loss, expected, atol=1e-5)


def test_gathers_at_teacher_indices_not_student_topk():
    """If we gathered at the student's own top-k, swapping teacher indices would not change KL."""
    s, y, tv, ti, _ = _fake_batch()
    out1 = kd_loss(s, y, tv, ti, alpha=1.0)
    ti_shuffled = ti.flip(-1)                         # same set, different order -> same KL
    out2 = kd_loss(s, y, tv.flip(-1), ti_shuffled, alpha=1.0)
    assert torch.isclose(out1.kl, out2.kl, atol=1e-5)
    ti_other = (ti + 7) % s.size(-1)                 # different vocab ids -> different KL
    out3 = kd_loss(s, y, tv, ti_other, alpha=1.0)
    assert not torch.isclose(out1.kl, out3.kl, atol=1e-3)


def test_masked_tokens_do_not_contribute():
    s, y, tv, ti, _ = _fake_batch()
    out_a = kd_loss(s, y, tv, ti, alpha=1.0)
    s2 = s.clone()
    s2[:, :2] += 100.0                                # positions whose *targets* are masked
    out_b = kd_loss(s2, y, tv, ti, alpha=1.0)
    assert torch.isclose(out_a.kl, out_b.kl, atol=1e-5)


def test_backward_flows():
    s, y, tv, ti, _ = _fake_batch()
    s.requires_grad_(True)
    out = kd_loss(s, y, tv, ti)
    out.loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_accepts_fp16_and_int32_inputs():
    s, y, tv, ti, _ = _fake_batch()
    out = kd_loss(s.half(), y, tv.half(), ti.to(torch.int32))
    assert torch.isfinite(out.loss)
