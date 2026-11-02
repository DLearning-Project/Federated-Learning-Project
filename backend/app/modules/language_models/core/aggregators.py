"""Federated LoRA aggregation: FedIT, exact weighted stacking, and FLoRIST.

Notation follows the FLoRIST paper (Ramesh & Dass, MLSys 2026). Client k
holds B_k (m x r_k) and A_k (r_k x n); its update is B_k @ A_k. Weights
w_k = n_k / N sum to one. The correct aggregate is

    delta_W = sum_k w_k B_k A_k                      (target_update)

FedIT averages B and A separately, which generally differs from delta_W.
Weighted stacking reproduces delta_W exactly. FLoRIST stacks, decomposes
each stack, and finds the SVD of delta_W through a compact matrix P without
forming delta_W (Eqs. 1-3, 6 and Algorithm 1).

This module follows Algorithm 1. The official code's stacking loop
(fed_utils/model_aggregation.py lines 43-75 at commit 100d373) runs a
`for ... else` block for every client, which stacks client 0 twice. See
docs/language-models/verification-notes.md, item 6.

Pure NumPy, float64, no web framework imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Relative tolerance when comparing cumulative energy with tau, so that
# tau = 1 is reached despite floating-point rounding in the cumulative sum.
ENERGY_RTOL = 1e-12

Matrices = Sequence[np.ndarray]


def normalize_weights(sample_counts: Sequence[float]) -> np.ndarray:
    """Return w_k = n_k / N. Counts must be finite, non-negative, and not all zero."""
    n = np.asarray(sample_counts, dtype=np.float64)
    if n.ndim != 1 or n.size == 0:
        raise ValueError("sample_counts must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("sample_counts must be finite and non-negative")
    total = n.sum()
    if total <= 0:
        raise ValueError("sample_counts must not all be zero")
    return n / total


def _check(Bs: Matrices, As: Matrices, w: Sequence[float]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Validate client factors and weights; return float64 copies."""
    if len(Bs) == 0 or len(Bs) != len(As) or len(As) != len(w):
        raise ValueError("Bs, As and w must be non-empty and the same length")
    B_list = [np.asarray(B, dtype=np.float64) for B in Bs]
    A_list = [np.asarray(A, dtype=np.float64) for A in As]
    for k, (B, A) in enumerate(zip(B_list, A_list)):
        if B.ndim != 2 or A.ndim != 2:
            raise ValueError(f"client {k}: B and A must be 2-D")
    m, n = B_list[0].shape[0], A_list[0].shape[1]
    for k, (B, A) in enumerate(zip(B_list, A_list)):
        if B.shape[1] != A.shape[0]:
            raise ValueError(f"client {k}: B is {B.shape} but A is {A.shape}; ranks differ")
        if B.shape[0] != m or A.shape[1] != n:
            raise ValueError(f"client {k}: all clients must share m = {m} and n = {n}")
        if B.shape[1] < 1:
            raise ValueError(f"client {k}: rank must be at least 1")
    weights = np.asarray(w, dtype=np.float64)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("w must be a 1-D sequence of finite, non-negative numbers")
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-9):
        raise ValueError("w must sum to 1; use normalize_weights")
    return B_list, A_list, weights


def target_update(Bs: Matrices, As: Matrices, w: Sequence[float]) -> np.ndarray:
    """delta_W = sum_k w_k B_k A_k, the aggregate every method should reproduce."""
    B_list, A_list, weights = _check(Bs, As, w)
    return sum(wk * (B @ A) for wk, B, A in zip(weights, B_list, A_list))


def fedit(Bs: Matrices, As: Matrices, w: Sequence[float]) -> np.ndarray:
    """FedIT aggregate (sum_k w_k B_k)(sum_k w_k A_k). Equal client ranks only."""
    B_list, A_list, weights = _check(Bs, As, w)
    ranks = {B.shape[1] for B in B_list}
    if len(ranks) != 1:
        raise ValueError(f"FedIT supports equal client ranks only; got ranks {sorted(ranks)}")
    B_avg = sum(wk * B for wk, B in zip(weights, B_list))
    A_avg = sum(wk * A for wk, A in zip(weights, A_list))
    return B_avg @ A_avg


def exact_stack(Bs: Matrices, As: Matrices, w: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Weighted stacking: B_stack = [B_1 ... B_K] (m x r), A_stack = [w_1 A_1; ...; w_K A_K] (r x n).

    B_stack @ A_stack equals target_update exactly; r = sum_k r_k.
    """
    B_list, A_list, weights = _check(Bs, As, w)
    B_stack = np.concatenate(B_list, axis=1)
    A_stack = np.concatenate([wk * A for wk, A in zip(weights, A_list)], axis=0)
    return B_stack, A_stack


@dataclass(frozen=True)
class FloristResult:
    """Output of florist(). Shapes use r = sum_k r_k, q = min(m, r), s = min(r, n)."""

    B_g: np.ndarray  # m x p
    A_g: np.ndarray  # p x n
    p: int
    tau: float
    singular_values: np.ndarray  # S_P, length min(q, s); equal to the singular values of delta_W
    cumulative_energy: np.ndarray  # same length; 0 where the aggregate is all zero
    B_stack: np.ndarray  # m x r
    A_stack: np.ndarray  # r x n
    U_B: np.ndarray  # m x q
    S_B: np.ndarray  # length q
    Vt_B: np.ndarray  # q x r
    U_A: np.ndarray  # r x s
    S_A: np.ndarray  # length s
    Vt_A: np.ndarray  # s x n
    Q: np.ndarray  # q x s
    P: np.ndarray  # q x s
    U_P: np.ndarray  # q x min(q, s)
    Vt_P: np.ndarray  # min(q, s) x s


def select_rank(singular_values: np.ndarray, tau: float) -> tuple[int, np.ndarray]:
    """Smallest p with sum_{i<=p} s_i^2 / sum_i s_i^2 >= tau (Eq. 6).

    Returns (p, cumulative_energy). An all-zero spectrum gives p = 0 and a zero
    cumulative energy, because there is nothing to keep.
    """
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must be in (0, 1]")
    energy = np.asarray(singular_values, dtype=np.float64) ** 2
    total = energy.sum()
    if total <= 0.0:
        return 0, np.zeros_like(energy)
    cumulative = np.cumsum(energy) / total
    reached = np.nonzero(cumulative >= tau * (1.0 - ENERGY_RTOL))[0]
    p = int(reached[0]) + 1 if reached.size else int(energy.size)
    return p, cumulative


def florist(Bs: Matrices, As: Matrices, w: Sequence[float], tau: float) -> FloristResult:
    """FLoRIST server aggregation (Eqs. 1-3 and 6, Algorithm 1).

    1. Weighted stacking.
    2. Reduced SVD of each stack: B_stack = U_B S_B Vt_B, A_stack = U_A S_A Vt_A.
    3. Q = Vt_B U_A and P = S_B Q S_A, shape min(m, r) x min(r, n).
    4. SVD of P = U_P S_P Vt_P; S_P are the singular values of delta_W.
    5. Keep the smallest p reaching energy tau.
    6. B_g = (U_B U_P)[:, :p] S_P[:p], A_g = (Vt_P Vt_A)[:p, :].
    """
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must be in (0, 1]")
    B_stack, A_stack = exact_stack(Bs, As, w)

    U_B, S_B, Vt_B = np.linalg.svd(B_stack, full_matrices=False)
    U_A, S_A, Vt_A = np.linalg.svd(A_stack, full_matrices=False)
    Q = Vt_B @ U_A
    P = (S_B[:, None] * Q) * S_A[None, :]
    U_P, S_P, Vt_P = np.linalg.svd(P, full_matrices=False)

    p, cumulative = select_rank(S_P, tau)
    B_g = (U_B @ U_P[:, :p]) * S_P[None, :p]
    A_g = Vt_P[:p, :] @ Vt_A

    return FloristResult(
        B_g=B_g,
        A_g=A_g,
        p=p,
        tau=float(tau),
        singular_values=S_P,
        cumulative_energy=cumulative,
        B_stack=B_stack,
        A_stack=A_stack,
        U_B=U_B,
        S_B=S_B,
        Vt_B=Vt_B,
        U_A=U_A,
        S_A=S_A,
        Vt_A=Vt_A,
        Q=Q,
        P=P,
        U_P=U_P,
        Vt_P=Vt_P,
    )
