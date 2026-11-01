"""Tests for the language-models aggregation engine (FedIT, exact stacking, FLoRIST)."""

import numpy as np
import pytest

from app.modules.language_models.core.aggregators import (
    exact_stack,
    fedit,
    florist,
    normalize_weights,
    select_rank,
    target_update,
)

ATOL = 1e-10


def random_clients(rng, m, n, ranks, scale=1.0):
    Bs = [scale * rng.normal(size=(m, r)) for r in ranks]
    As = [scale * rng.normal(size=(r, n)) for r in ranks]
    w = normalize_weights(rng.integers(1, 100, size=len(ranks)))
    return Bs, As, w


def fro(M):
    return float(np.linalg.norm(M, "fro"))


def truncate(res, q):
    """Rank-q global adapters from the same FLoRIST decomposition."""
    U = res.U_B @ res.U_P
    return (U[:, :q] * res.singular_values[None, :q]) @ (res.Vt_P[:q, :] @ res.Vt_A)


# --- Hand-calculated example ---------------------------------------------------
# Two rank-1 clients with counts 3 and 1, so w = (0.75, 0.25):
#   B_1 = e_1, A_1 = e_1^T and B_2 = e_2, A_2 = e_2^T
#   delta_W = diag(0.75, 0.25); singular values 0.75 and 0.25
#   energy of the first component: 0.5625 / 0.625 = 0.9
#   FedIT: (0.75 e_1 + 0.25 e_2)(0.75 e_1 + 0.25 e_2)^T = [[0.5625, 0.1875], [0.1875, 0.0625]]
#   FedIT error matrix [[-0.1875, 0.1875], [0.1875, -0.1875]], Frobenius norm 0.375

E1, E2 = np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])
HAND = ([E1, E2], [E1.T, E2.T], normalize_weights([3, 1]))


def test_hand_example_target_and_spectrum():
    Bs, As, w = HAND
    np.testing.assert_allclose(w, [0.75, 0.25])
    np.testing.assert_allclose(target_update(Bs, As, w), np.diag([0.75, 0.25]), atol=ATOL)
    res = florist(Bs, As, w, tau=0.9)
    np.testing.assert_allclose(res.singular_values, [0.75, 0.25], atol=ATOL)
    np.testing.assert_allclose(res.cumulative_energy, [0.9, 1.0], atol=ATOL)
    assert res.p == 1  # 0.9 reaches tau = 0.9 exactly
    assert florist(Bs, As, w, tau=0.91).p == 2


def test_hand_example_fedit_counterexample():
    """Selected counterexample: differing factors give nonzero FedIT error."""
    Bs, As, w = HAND
    expected = np.array([[0.5625, 0.1875], [0.1875, 0.0625]])
    np.testing.assert_allclose(fedit(Bs, As, w), expected, atol=ATOL)
    assert fro(fedit(Bs, As, w) - target_update(Bs, As, w)) == pytest.approx(0.375)


# --- Step 2 tests from README_FL ------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_florist_tau_one_reproduces_target(seed):
    rng = np.random.default_rng(seed)
    Bs, As, w = random_clients(rng, 40, 30, [4, 8, 2])
    res = florist(Bs, As, w, tau=1.0)
    np.testing.assert_allclose(res.B_g @ res.A_g, target_update(Bs, As, w), atol=ATOL)


@pytest.mark.parametrize("seed", range(5))
def test_singular_values_of_P_match_direct_svd(seed):
    rng = np.random.default_rng(seed)
    Bs, As, w = random_clients(rng, 25, 35, [3, 5, 1, 4])
    res = florist(Bs, As, w, tau=0.9)
    direct = np.linalg.svd(target_update(Bs, As, w), compute_uv=False)
    k = res.singular_values.size
    np.testing.assert_allclose(res.singular_values, direct[:k], atol=ATOL)
    assert np.all(direct[k:] < ATOL)  # the rest of delta_W's spectrum is zero


@pytest.mark.parametrize("seed", range(5))
def test_weighted_stacking_reproduces_target(seed):
    rng = np.random.default_rng(seed)
    Bs, As, w = random_clients(rng, 12, 9, [1, 3, 2])
    B_stack, A_stack = exact_stack(Bs, As, w)
    assert B_stack.shape == (12, 6) and A_stack.shape == (6, 9)
    np.testing.assert_allclose(B_stack @ A_stack, target_update(Bs, As, w), atol=ATOL)


def test_fedit_identical_factors_zero_error():
    rng = np.random.default_rng(0)
    B, A = rng.normal(size=(10, 3)), rng.normal(size=(3, 7))
    w = normalize_weights([5, 1, 2])
    Bs, As = [B, B, B], [A, A, A]
    np.testing.assert_allclose(fedit(Bs, As, w), target_update(Bs, As, w), atol=ATOL)


@pytest.mark.parametrize("seed", range(3))
def test_fedit_differing_factors_nonzero_error(seed):
    rng = np.random.default_rng(seed)
    Bs, As, w = random_clients(rng, 10, 8, [3, 3, 3])
    assert fro(fedit(Bs, As, w) - target_update(Bs, As, w)) > 1e-3


def test_fedit_shared_B_zero_error():
    """Clients differ (different A_k) yet FedIT is exact: (B)(sum w A) = sum w B A."""
    rng = np.random.default_rng(1)
    B = rng.normal(size=(10, 3))
    As = [rng.normal(size=(3, 7)) for _ in range(3)]
    w = normalize_weights([2, 3, 5])
    Bs = [B, B, B]
    assert fro(target_update(Bs, As, w) - B @ As[0]) > 1e-3  # clients really differ
    np.testing.assert_allclose(fedit(Bs, As, w), target_update(Bs, As, w), atol=ATOL)


def test_fedit_identical_products_different_factors_nonzero_error():
    """B_2 A_2 = (2B)(A/2) = B_1 A_1, but FedIT gives (1.5 B)(0.75 A) = 1.125 BA."""
    rng = np.random.default_rng(2)
    B, A = rng.normal(size=(6, 2)), rng.normal(size=(2, 5))
    Bs, As, w = [B, 2 * B], [A, A / 2], normalize_weights([1, 1])
    np.testing.assert_allclose(Bs[0] @ As[0], Bs[1] @ As[1], atol=ATOL)
    np.testing.assert_allclose(target_update(Bs, As, w), B @ A, atol=ATOL)
    np.testing.assert_allclose(fedit(Bs, As, w), 1.125 * (B @ A), atol=ATOL)


@pytest.mark.parametrize("seed", range(3))
def test_selected_rank_nondecreasing_in_tau(seed):
    rng = np.random.default_rng(seed)
    Bs, As, w = random_clients(rng, 30, 20, [4, 4, 8])
    ps = [florist(Bs, As, w, tau=t).p for t in np.linspace(0.05, 1.0, 40)]
    assert all(a <= b for a, b in zip(ps, ps[1:]))
    assert ps[-1] == np.linalg.matrix_rank(target_update(Bs, As, w))


@pytest.mark.parametrize("seed", range(3))
def test_reconstruction_error_nonincreasing_in_rank(seed):
    rng = np.random.default_rng(seed)
    Bs, As, w = random_clients(rng, 30, 20, [4, 4, 8])
    target = target_update(Bs, As, w)
    res = florist(Bs, As, w, tau=1.0)
    errors = [fro(target - truncate(res, q)) for q in range(res.singular_values.size + 1)]
    assert all(b <= a + ATOL for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize("seed", range(10))
def test_eckart_young_bound_eq5(seed):
    rng = np.random.default_rng(seed)
    m, n = rng.integers(5, 40, size=2)
    ranks = rng.integers(1, 6, size=rng.integers(1, 6))
    Bs, As, w = random_clients(rng, m, n, ranks)
    target = target_update(Bs, As, w)
    for tau in (0.5, 0.8, 0.9, 0.99):
        res = florist(Bs, As, w, tau=tau)
        tail = float(np.sqrt(np.sum(res.singular_values[res.p:] ** 2)))
        err = fro(target - res.B_g @ res.A_g)
        assert err <= tail + 1e-9
        assert err == pytest.approx(tail, abs=1e-9)  # truncated SVD attains the bound


@pytest.mark.parametrize(
    "m, n, ranks",
    [
        (3, 20, [2, 2]),  # r = 4 > m
        (20, 3, [2, 2]),  # r = 4 > n
        (3, 4, [2, 3, 4]),  # r = 9 > m and r > n
    ],
)
def test_shapes_when_stacked_rank_exceeds_dims(m, n, ranks):
    rng = np.random.default_rng(3)
    Bs, As, w = random_clients(rng, m, n, ranks)
    r = sum(ranks)
    res = florist(Bs, As, w, tau=1.0)
    q, s = min(m, r), min(r, n)
    assert res.B_stack.shape == (m, r) and res.A_stack.shape == (r, n)
    assert res.Q.shape == (q, s) and res.P.shape == (q, s)
    assert res.B_g.shape == (m, res.p) and res.A_g.shape == (res.p, n)
    np.testing.assert_allclose(res.B_g @ res.A_g, target_update(Bs, As, w), atol=ATOL)


def test_Q_orthogonal_when_stacked_rank_small():
    rng = np.random.default_rng(4)
    Bs, As, w = random_clients(rng, 30, 25, [3, 4])
    res = florist(Bs, As, w, tau=0.9)
    assert res.Q.shape == (7, 7)
    np.testing.assert_allclose(res.Q.T @ res.Q, np.eye(7), atol=ATOL)


def test_all_zero_aggregate_gives_rank_zero():
    Bs = [np.zeros((5, 2)), np.zeros((5, 3))]
    As = [np.ones((2, 4)), np.ones((3, 4))]  # B = 0, as in round 1 before training
    res = florist(Bs, As, normalize_weights([1, 1]), tau=0.9)
    assert res.p == 0
    assert res.B_g.shape == (5, 0) and res.A_g.shape == (0, 4)
    np.testing.assert_allclose(res.B_g @ res.A_g, np.zeros((5, 4)))
    assert np.all(res.cumulative_energy == 0)


# --- Input validation -------------------------------------------------------------


def test_fedit_rejects_unequal_ranks():
    rng = np.random.default_rng(5)
    Bs, As, w = random_clients(rng, 6, 5, [2, 3])
    with pytest.raises(ValueError, match="equal client ranks"):
        fedit(Bs, As, w)


@pytest.mark.parametrize("tau", [0.0, -0.1, 1.01])
def test_tau_out_of_range_rejected(tau):
    Bs, As, w = HAND
    with pytest.raises(ValueError, match="tau"):
        florist(Bs, As, w, tau=tau)
    with pytest.raises(ValueError, match="tau"):
        select_rank(np.array([1.0]), tau)


def test_invalid_inputs_rejected():
    Bs, As, _ = HAND
    with pytest.raises(ValueError, match="sum to 1"):
        target_update(Bs, As, [0.5, 0.6])
    with pytest.raises(ValueError, match="same length"):
        target_update(Bs, As[:1], [1.0])
    with pytest.raises(ValueError, match="ranks differ"):
        target_update([np.ones((2, 2))], [np.ones((3, 2))], [1.0])
    with pytest.raises(ValueError, match="share m"):
        target_update([np.ones((2, 1)), np.ones((3, 1))], [np.ones((1, 2)), np.ones((1, 2))], [0.5, 0.5])
    with pytest.raises(ValueError, match="not all be zero"):
        normalize_weights([0, 0])
    with pytest.raises(ValueError, match="non-negative"):
        normalize_weights([1, -1])
