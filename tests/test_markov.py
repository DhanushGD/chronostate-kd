import torch

from chronostate.markov import MarkovStateLayer


def test_init_is_sticky_and_row_stochastic():
    m = MarkovStateLayer(["code", "reasoning"], stickiness=0.8)
    assert m.is_row_stochastic()
    assert torch.allclose(m.T, torch.tensor([[0.8, 0.2], [0.2, 0.8]]))


def test_prior_uniform_before_first_state():
    m = MarkovStateLayer(["code", "reasoning", "general"])
    assert torch.allclose(m.prior(), torch.full((3,), 1 / 3))


def test_update_moves_row_toward_observed_transition_and_stays_stochastic():
    m = MarkovStateLayer(["code", "reasoning"], stickiness=0.8, ema=0.1)
    m.update(0)                     # first observation sets prev only
    assert m.n_updates == 0
    m.update(1)                     # code -> reasoning
    assert m.n_updates == 1
    assert torch.allclose(m.T[0], torch.tensor([0.72, 0.28]))
    assert m.is_row_stochastic()
    assert torch.allclose(m.T[1], torch.tensor([0.2, 0.8]))   # untouched row


def test_repeated_sticky_updates_increase_stickiness():
    m = MarkovStateLayer(["code", "reasoning"], stickiness=0.5, ema=0.2)
    m.update(0)
    for _ in range(20):
        m.update(0)
    assert m.T[0, 0] > 0.95
    assert m.is_row_stochastic()


def test_reset_forgets_state_but_keeps_T():
    m = MarkovStateLayer(["code", "reasoning"])
    m.update(0); m.update(1)
    T_before = m.T.clone()
    m.reset()
    assert m.prev_state is None
    assert torch.equal(m.T, T_before)
    assert torch.allclose(m.prior(), torch.full((2,), 0.5))


def test_custom_init_T_is_normalised():
    m = MarkovStateLayer(["a", "b"], init_T=torch.tensor([[3.0, 1.0], [1.0, 1.0]]))
    assert torch.allclose(m.T, torch.tensor([[0.75, 0.25], [0.5, 0.5]]))
