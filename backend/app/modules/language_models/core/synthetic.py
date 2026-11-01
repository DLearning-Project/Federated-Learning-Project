"""Synthetic client adapters for the Simulator page (stand-ins for trained ones).

Until the toy experiment (build step 5) trains real adapters, each client's
adapter pair is built from a target update with the structure the toy
specification calls for:

    T_i = S + C_i
    S   = one shared low-rank shift (rank SHARED_RANK), the same for all clients
    C_i = a smaller client-specific component (rank CLIENT_RANK), fixed per client

Client i's adapters are the best rank-r_i factorization of T_i, split evenly
between the factors: B = U_r sqrt(S_r), A = sqrt(S_r) V_r^T. No training
happens; the FLoRIST aggregation applied to these adapters is computed live.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

M, N = 16, 32  # toy adapted matrix W_0 is 16 x 32 (16 classes x 32 features)
SHARED_RANK = 3
SHARED_STRENGTHS = (1.0, 0.6, 0.35)
CLIENT_RANK = 2
CLIENT_SCALE = 0.3

ASSUMPTIONS = [
    "The adapted matrix is the toy model's W_0 (16 x 32), not an LLM layer.",
    "Client adapters are SYNTHETIC stand-ins for trained ones: each client's update is a shared rank-3 shift plus a smaller rank-2 client-specific part, factored at the client's rank.",
    "No local training happens yet; the toy experiment (build step 5) will replace these adapters.",
    "The aggregation (FedIT, exact weighted stacking, FLoRIST) is computed live by the backend.",
]


def _orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


def shared_shift(seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 1_000_003])
    U = _orthonormal(rng, M, SHARED_RANK)
    V = _orthonormal(rng, N, SHARED_RANK)
    return (U * np.asarray(SHARED_STRENGTHS)) @ V.T


def client_target(seed: int, client_id: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 2_000_029, client_id])
    U = _orthonormal(rng, M, CLIENT_RANK)
    V = _orthonormal(rng, N, CLIENT_RANK)
    strengths = CLIENT_SCALE * rng.uniform(0.5, 1.0, size=CLIENT_RANK)
    return shared_shift(seed) + (U * strengths) @ V.T


def synthetic_adapters(seed: int, client_ids: Sequence[int], ranks: Sequence[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (Bs, As): client i gets the best rank-r_i factorization of T_i."""
    if len(client_ids) != len(ranks):
        raise ValueError("client_ids and ranks must have the same length")
    Bs, As = [], []
    for cid, r in zip(client_ids, ranks):
        if r < 1 or r > min(M, N):
            raise ValueError(f"rank {r} is outside 1..{min(M, N)}")
        U, s, Vt = np.linalg.svd(client_target(seed, int(cid)), full_matrices=False)
        root = np.sqrt(s[:r])
        Bs.append(U[:, :r] * root)
        As.append(root[:, None] * Vt[:r, :])
    return Bs, As
