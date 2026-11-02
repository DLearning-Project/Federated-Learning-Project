"""Client-side rank matching from Algorithm 1 of the FLoRIST paper.

Every client receives the same rank-p global adapters (B_g, A_g) and fits
them to its own rank r_k before local training:

- p < r_k: zero-pad BOTH factors, B with r_k - p zero columns and A with
  r_k - p zero rows. B_k @ A_k is unchanged.
- p > r_k: truncate BOTH factors to the first r_k components. FLoRIST orders
  components by singular value, so this keeps the strongest directions.
- p = r_k: use them as they are.

When both padded factors are zero, the padded components receive zero
gradient (dL/dB = G A^T, dL/dA = B^T G), so they stay inactive during that
round's local training under ordinary gradient-based optimization. A client
with r_k > p therefore ends the round with an adapter of rank at most p. The
cap is per round: independently trained clients can aggregate into a
higher-rank update, so a later round's p may be larger.

Edge case: if p = 0 (all-zero aggregate), both factors are entirely zero and
the client cannot learn anything in that round under gradient descent.
"""

from __future__ import annotations

import numpy as np


def match_rank(B_g: np.ndarray, A_g: np.ndarray, r_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (B_k, A_k) of rank r_k from global adapters (B_g: m x p, A_g: p x n)."""
    B_g = np.asarray(B_g, dtype=np.float64)
    A_g = np.asarray(A_g, dtype=np.float64)
    if B_g.ndim != 2 or A_g.ndim != 2 or B_g.shape[1] != A_g.shape[0]:
        raise ValueError(f"B_g {B_g.shape} and A_g {A_g.shape} must be m x p and p x n")
    if int(r_k) != r_k or r_k < 1:
        raise ValueError("r_k must be a positive integer")
    r_k = int(r_k)
    m, p = B_g.shape
    n = A_g.shape[1]
    if p < r_k:
        B_k = np.hstack([B_g, np.zeros((m, r_k - p))])
        A_k = np.vstack([A_g, np.zeros((r_k - p, n))])
    elif p > r_k:
        B_k = B_g[:, :r_k].copy()
        A_k = A_g[:r_k, :].copy()
    else:
        B_k, A_k = B_g.copy(), A_g.copy()
    return B_k, A_k
