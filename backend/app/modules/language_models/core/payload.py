"""Payload accounting (bytes) and estimated server cost for one adapted matrix.

All values are COMPUTED from matrix shapes and precision; they are not the
paper's reported numbers (Table 3 uses a different convention, see
docs/language-models/verification-notes.md item 4). Upload and download are
kept separate. Server cost is an ESTIMATE, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

PAYLOAD_ASSUMPTIONS = [
    r"One adapted weight matrix $W$ ($m \times n$); each adapter pair has $r(m + n)$ numbers.",
    r"Every participating client uploads its own $B_k$ and $A_k$: $r_k(m + n)$ numbers.",
    "Every participating client downloads the same global adapters once per round.",
    r"FedIT downloads the averaged pair at the common rank $r$: $r(m + n)$ numbers.",
    r"Exact weighted stacking (reference) downloads the stack: $\left(\sum_k r_k\right)(m + n)$ numbers.",
    r"FLoRIST downloads the thresholded pair: $p(m + n)$ numbers.",
    "Bytes = numbers × bytes per number. 1 MB = 1,000,000 bytes (decimal).",
    "Not counted: the one-time base model download, the aggregation weights, protocol overhead, compression or encryption.",
]

SERVER_COST_ASSUMPTIONS = [
    "Estimated operation counts, not measurements.",
    r"A matrix product $(a \times b)(b \times c)$ counts $2abc$ operations.",
    r"A thin SVD of an $a \times b$ matrix counts $ab \min(a, b)$ operations (leading order only; real SVDs cost a constant multiple more).",
    r"FedIT: weighted sums of the $K$ pairs of factors.",
    r"FLoRIST: SVDs of both stacks, $Q$, $P$, SVD of $P$, and building $B_g$ and $A_g$ (Eqs. 1–3).",
    r"Direct reference: forming $\Delta W$ from the stack and taking its SVD, for comparison only.",
]


@dataclass(frozen=True)
class MethodPayload:
    method: str
    upload_per_client_bytes: list[int]
    upload_total_bytes: int
    download_rank: int
    download_per_client_bytes: int
    download_total_bytes: int


@dataclass(frozen=True)
class PayloadReport:
    methods: dict[str, MethodPayload]
    assumptions: list[str] = field(default_factory=lambda: list(PAYLOAD_ASSUMPTIONS))


def _check_dims(m: int, n: int, ranks: Sequence[int], bytes_per_param: int) -> None:
    if m < 1 or n < 1:
        raise ValueError("m and n must be positive")
    if len(ranks) == 0 or any(int(r) != r or r < 1 for r in ranks):
        raise ValueError("ranks must be a non-empty list of positive integers")
    if bytes_per_param not in (1, 2, 4, 8):
        raise ValueError("bytes_per_param must be 1, 2, 4 or 8")


def payload_per_round(m: int, n: int, ranks: Sequence[int], p: int, bytes_per_param: int) -> PayloadReport:
    """Upload and download bytes for one round, per method.

    ranks: each participating client's rank r_k. p: FLoRIST's selected rank.
    FedIT is included only when all ranks are equal (first-release rule).
    """
    _check_dims(m, n, ranks, bytes_per_param)
    if int(p) != p or p < 0:
        raise ValueError("p must be a non-negative integer")
    width = m + n
    K = len(ranks)
    upload = [int(r) * width * bytes_per_param for r in ranks]
    upload_total = sum(upload)

    download_ranks: dict[str, int] = {"exact_stack": int(sum(ranks)), "florist": int(p)}
    if len(set(ranks)) == 1:
        download_ranks = {"fedit": int(ranks[0]), **download_ranks}

    methods = {}
    for method, rank in download_ranks.items():
        per_client = rank * width * bytes_per_param
        methods[method] = MethodPayload(
            method=method,
            upload_per_client_bytes=list(upload),
            upload_total_bytes=upload_total,
            download_rank=rank,
            download_per_client_bytes=per_client,
            download_total_bytes=per_client * K,
        )
    return PayloadReport(methods=methods)


@dataclass(frozen=True)
class ServerCostEstimate:
    fedit_ops: int | None
    florist_ops: int
    direct_svd_ops: int
    stacked_rank: int
    stacked_rank_below_min_dim: bool
    assumptions: list[str] = field(default_factory=lambda: list(SERVER_COST_ASSUMPTIONS))


def _svd_ops(a: int, b: int) -> int:
    return a * b * min(a, b)


def _matmul_ops(a: int, b: int, c: int) -> int:
    return 2 * a * b * c


def estimate_server_ops(m: int, n: int, ranks: Sequence[int], p: int) -> ServerCostEstimate:
    """Estimated server operation counts for one adapted matrix in one round."""
    _check_dims(m, n, ranks, 8)
    if int(p) != p or p < 0:
        raise ValueError("p must be a non-negative integer")
    r = int(sum(ranks))
    q, s = min(m, r), min(r, n)  # shapes of U_B (m x q), U_A (r x s), P (q x s)
    if p > min(q, s):
        raise ValueError(f"p = {p} exceeds the largest possible rank {min(q, s)}")
    florist_ops = (
        r * n  # weight the A_k rows
        + _svd_ops(m, r)
        + _svd_ops(r, n)
        + _matmul_ops(q, r, s)  # Q = Vt_B U_A
        + 2 * q * s  # P = S_B Q S_A (row and column scaling)
        + _svd_ops(q, s)
        + _matmul_ops(m, q, p) + m * p  # B_g = (U_B U_P[:, :p]) S_P
        + _matmul_ops(p, s, n)  # A_g = Vt_P[:p] Vt_A
    )
    # FedIT: one multiply and one add per entry of every client's B_k and A_k.
    fedit_ops = 2 * r * (m + n) if len(set(ranks)) == 1 else None
    direct_ops = _matmul_ops(m, r, n) + _svd_ops(m, n)
    return ServerCostEstimate(
        fedit_ops=fedit_ops,
        florist_ops=florist_ops,
        direct_svd_ops=direct_ops,
        stacked_rank=r,
        stacked_rank_below_min_dim=r < min(m, n),
    )
