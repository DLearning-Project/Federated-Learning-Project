"""Step 3: FLoRIST intermediate states, rank matching, payload accounting."""

import numpy as np
import pytest

from app.modules.language_models.core.aggregators import florist, normalize_weights, target_update
from app.modules.language_models.core.florist_steps import florist_steps
from app.modules.language_models.core.payload import (
    PAYLOAD_ASSUMPTIONS,
    estimate_server_ops,
    payload_per_round,
)
from app.modules.language_models.core.rank_matching import match_rank

ATOL = 1e-10


def random_clients(rng, m, n, ranks):
    Bs = [rng.normal(size=(m, r)) for r in ranks]
    As = [rng.normal(size=(r, n)) for r in ranks]
    return Bs, As, normalize_weights(rng.integers(1, 100, size=len(ranks)))


# --- Local training used by the gradient and multi-round tests -------------------
# Least-squares loss L(B, A) = 0.5 * ||B A - T||_F^2, so with G = B A - T:
#   dL/dB = G A^T and dL/dA = B^T G.


def grads(B, A, T):
    G = B @ A - T
    return G @ A.T, B.T @ G


def train_gd(B, A, T, lr=0.05, steps=400):
    B, A = B.copy(), A.copy()
    for _ in range(steps):
        dB, dA = grads(B, A, T)
        B -= lr * dB
        A -= lr * dA
    return B, A


def train_adam(B, A, T, lr=0.01, steps=400, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    """AdamW with fresh moment estimates, as each round's new optimizer would have."""
    params = [B.copy(), A.copy()]
    m = [np.zeros_like(x) for x in params]
    v = [np.zeros_like(x) for x in params]
    for t in range(1, steps + 1):
        for i, g in enumerate(grads(params[0], params[1], T)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            update = (m[i] / (1 - beta1**t)) / (np.sqrt(v[i] / (1 - beta2**t)) + eps)
            params[i] = params[i] - lr * (update + weight_decay * params[i])
    return params[0], params[1]


# --- Rank matching ------------------------------------------------------------------


def test_zero_padding_leaves_product_unchanged():
    rng = np.random.default_rng(0)
    B_g, A_g = rng.normal(size=(7, 2)), rng.normal(size=(2, 5))
    B_k, A_k = match_rank(B_g, A_g, 4)
    assert B_k.shape == (7, 4) and A_k.shape == (4, 5)
    assert np.all(B_k[:, 2:] == 0) and np.all(A_k[2:, :] == 0)  # both factors padded
    np.testing.assert_allclose(B_k @ A_k, B_g @ A_g, atol=ATOL)


def test_truncation_keeps_strongest_components():
    rng = np.random.default_rng(1)
    Bs, As, w = random_clients(rng, 12, 10, [4, 4])
    res = florist(Bs, As, w, tau=1.0)
    assert res.p > 3
    B_k, A_k = match_rank(res.B_g, res.A_g, 3)
    assert B_k.shape == (12, 3) and A_k.shape == (3, 10)
    # The first 3 FLoRIST components are the best rank-3 approximation of delta_W.
    target = target_update(Bs, As, w)
    tail = np.sqrt(np.sum(res.singular_values[3:] ** 2))
    assert np.linalg.norm(target - B_k @ A_k, "fro") == pytest.approx(tail, abs=1e-9)


def test_equal_rank_is_unchanged_copy():
    B_g, A_g = np.ones((3, 2)), np.ones((2, 4))
    B_k, A_k = match_rank(B_g, A_g, 2)
    np.testing.assert_array_equal(B_k, B_g)
    B_k[0, 0] = 5.0
    assert B_g[0, 0] == 1.0  # a copy, not a view


def test_match_rank_rejects_bad_input():
    with pytest.raises(ValueError):
        match_rank(np.ones((3, 2)), np.ones((3, 4)), 2)
    with pytest.raises(ValueError):
        match_rank(np.ones((3, 2)), np.ones((2, 4)), 0)


# --- Gradient check: padded components are inactive within one round --------------


@pytest.mark.parametrize("trainer", [train_gd, train_adam], ids=["sgd", "adamw"])
def test_padded_components_get_zero_gradient_and_stay_zero(trainer):
    rng = np.random.default_rng(2)
    m, n, p, r_k = 8, 6, 2, 5
    B_g, A_g = rng.normal(size=(m, p)), rng.normal(size=(p, n))
    T = rng.normal(size=(m, n))  # full-rank target: the client would like more rank
    B_k, A_k = match_rank(B_g, A_g, r_k)

    dB, dA = grads(B_k, A_k, T)
    assert np.all(dB[:, p:] == 0) and np.all(dA[p:, :] == 0)

    B_new, A_new = trainer(B_k, A_k, T)
    assert np.all(B_new[:, p:] == 0) and np.all(A_new[p:, :] == 0)
    assert np.linalg.matrix_rank(B_new @ A_new) <= p
    assert not np.allclose(B_new[:, :p], B_g)  # the active part did train


def test_nonzero_padding_does_activate():
    """Documents the qualifier: if a padded factor is nonzero, components can activate."""
    rng = np.random.default_rng(3)
    m, n, p, r_k = 8, 6, 2, 5
    B_g, A_g = rng.normal(size=(m, p)), rng.normal(size=(p, n))
    T = rng.normal(size=(m, n))
    B_k, A_k = match_rank(B_g, A_g, r_k)
    A_k[p:, :] = 0.01 * rng.normal(size=(r_k - p, n))  # random padding of A only
    B_new, A_new = train_gd(B_k, A_k, T)
    assert np.any(B_new[:, p:] != 0)
    assert np.linalg.matrix_rank(B_new @ A_new) > p


# --- Multi-round: the cap is per round, so p can grow ------------------------------


def test_selected_rank_can_increase_between_rounds():
    rng = np.random.default_rng(4)
    m, n, r_k = 8, 8, 4
    # Round t starts from a rank-1 global adapter.
    B_g, A_g = 0.1 * rng.normal(size=(m, 1)), 0.1 * rng.normal(size=(1, n))
    # Three clients whose local data point in different rank-1 directions.
    targets = [np.outer(rng.normal(size=m), rng.normal(size=n)) for _ in range(3)]

    Bs, As = [], []
    for T in targets:
        B_k, A_k = match_rank(B_g, A_g, r_k)
        B_k, A_k = train_gd(B_k, A_k, T, lr=0.02, steps=2000)
        assert np.linalg.matrix_rank(B_k @ A_k, tol=1e-8) <= 1  # capped at p = 1 this round
        Bs.append(B_k)
        As.append(A_k)

    next_round = florist(Bs, As, normalize_weights([1, 1, 1]), tau=0.99)
    assert next_round.p == 3  # the aggregate of rank-1 updates has rank 3


# --- Named states and errors ----------------------------------------------------------

EXPECTED_STATES = [
    "B_1", "A_1", "B_2", "A_2", "B_3", "A_3",
    "B_stack", "A_stack", "U_B", "S_B", "Vt_B", "U_A", "S_A", "Vt_A",
    "Q", "P", "U_P", "S_P", "Vt_P", "cumulative_energy", "B_g", "A_g",
]


def test_florist_steps_names_shapes_and_errors():
    rng = np.random.default_rng(5)
    m, n, ranks = 16, 32, [2, 2, 2]
    Bs, As, w = random_clients(rng, m, n, ranks)
    steps = florist_steps(Bs, As, w, tau=0.9)

    assert [s.name for s in steps.states] == EXPECTED_STATES
    for s in steps.states:
        assert s.shape == s.value.shape and s.description
    r = sum(ranks)
    assert steps.state("B_stack").shape == (m, r)
    assert steps.state("A_stack").shape == (r, n)
    assert steps.state("P").shape == (min(m, r), min(r, n))
    assert steps.state("B_g").shape == (m, steps.p)
    assert steps.state("A_g").shape == (steps.p, n)
    assert steps.delta_w_shape == (m, n)

    assert steps.aggregation_error["exact_stack"].absolute < 1e-9
    assert steps.aggregation_error["fedit"].absolute > 1e-3
    assert steps.reconstruction_error.absolute == pytest.approx(steps.eq5_bound, abs=1e-9)
    assert 0 <= steps.reconstruction_error.relative < 1


def test_florist_steps_fedit_error_absent_for_unequal_ranks():
    rng = np.random.default_rng(6)
    Bs, As, w = random_clients(rng, 10, 9, [1, 2, 4])
    steps = florist_steps(Bs, As, w, tau=0.9)
    assert steps.aggregation_error["fedit"] is None
    assert [s.name for s in steps.states][:2] == ["B_1", "A_1"]


def test_florist_steps_zero_aggregate_has_no_relative_error():
    Bs, As = [np.zeros((4, 1)), np.zeros((4, 1))], [np.ones((1, 3)), np.ones((1, 3))]
    steps = florist_steps(Bs, As, normalize_weights([1, 1]), tau=0.9)
    assert steps.p == 0
    assert steps.reconstruction_error.absolute == 0 and steps.reconstruction_error.relative is None


# --- Payload accounting -----------------------------------------------------------------


def test_payload_hand_calculation():
    # m = 16, n = 32 (m + n = 48), three clients at rank 2, p = 3, fp32 (4 bytes).
    report = payload_per_round(16, 32, [2, 2, 2], p=3, bytes_per_param=4)
    fedit, stack, flor = report.methods["fedit"], report.methods["exact_stack"], report.methods["florist"]

    for method in (fedit, stack, flor):
        assert method.upload_per_client_bytes == [384, 384, 384]  # 2 x 48 x 4
        assert method.upload_total_bytes == 1152
    assert (fedit.download_rank, fedit.download_per_client_bytes, fedit.download_total_bytes) == (2, 384, 1152)
    assert (stack.download_rank, stack.download_per_client_bytes, stack.download_total_bytes) == (6, 1152, 3456)
    assert (flor.download_rank, flor.download_per_client_bytes, flor.download_total_bytes) == (3, 576, 1728)
    assert report.assumptions == PAYLOAD_ASSUMPTIONS


def test_payload_heterogeneous_ranks_omit_fedit():
    report = payload_per_round(10, 20, [1, 2, 4], p=2, bytes_per_param=2)
    assert set(report.methods) == {"exact_stack", "florist"}
    assert report.methods["exact_stack"].upload_per_client_bytes == [60, 120, 240]
    assert report.methods["exact_stack"].download_total_bytes == 7 * 30 * 2 * 3


def test_payload_rejects_bad_input():
    with pytest.raises(ValueError):
        payload_per_round(10, 20, [], p=1, bytes_per_param=2)
    with pytest.raises(ValueError):
        payload_per_round(10, 20, [2], p=1, bytes_per_param=3)
    with pytest.raises(ValueError):
        payload_per_round(10, 20, [2], p=-1, bytes_per_param=2)


# --- Server cost estimate -------------------------------------------------------------------


def test_server_cost_estimate_hand_calculation():
    # m = 16, n = 32, ranks [2, 2, 2] (r = 6), p = 3; q = min(16, 6) = 6, s = min(6, 32) = 6.
    est = estimate_server_ops(16, 32, [2, 2, 2], p=3)
    expected = (
        6 * 32  # weight the A rows
        + 16 * 6 * 6  # SVD of B_stack
        + 6 * 32 * 6  # SVD of A_stack
        + 2 * 6 * 6 * 6  # Q
        + 2 * 6 * 6  # P
        + 6 * 6 * 6  # SVD of P
        + 2 * 16 * 6 * 3 + 16 * 3  # B_g
        + 2 * 3 * 6 * 32  # A_g
    )
    assert est.florist_ops == expected
    assert est.fedit_ops == 2 * 6 * 48
    assert est.direct_svd_ops == 2 * 16 * 6 * 32 + 16 * 32 * 16
    assert est.stacked_rank == 6 and est.stacked_rank_below_min_dim
    assert any("Estimated" in a for a in est.assumptions)


def test_server_cost_flags_when_compact_advantage_is_gone():
    est = estimate_server_ops(16, 32, [8, 8], p=4)
    assert est.stacked_rank == 16 and not est.stacked_rank_below_min_dim
    assert est.fedit_ops is not None
    assert estimate_server_ops(16, 32, [4, 8], p=4).fedit_ops is None
    with pytest.raises(ValueError):
        estimate_server_ops(16, 32, [2, 2], p=5)  # p cannot exceed min(m, r, n) = 4
