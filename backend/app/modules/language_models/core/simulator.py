"""One Simulator round: sample clients, weight them by data, aggregate live.

Per round t:
1. Partition the toy dataset across all clients with Dirichlet(alpha) labels.
2. Sample clients_per_round clients (seeded by seed and t).
3. Weight sampled clients by their sample counts (w_k = n_k / N).
4. Build their synthetic adapters (see synthetic.py) at their LoRA ranks.
5. Run FLoRIST, FedIT (equal ranks only) and exact stacking, with payload
   and estimated server cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aggregators import normalize_weights
from .florist_steps import FloristSteps, florist_steps
from .partition import dirichlet_label_counts
from .payload import PayloadReport, ServerCostEstimate, estimate_server_ops, payload_per_round
from .synthetic import ASSUMPTIONS as SYNTHETIC_ASSUMPTIONS
from .synthetic import M, N, synthetic_adapters

NUM_CLASSES = 16
SAMPLES_PER_CLASS = 60
HOMOGENEOUS_RANK = 4
# Heterogeneous mix over the toy ranks 1, 2, 4, heavy-tailed like the paper's
# (most clients small): by client id, 50% rank 1, 30% rank 2, 20% rank 4.
HETEROGENEOUS_PATTERN = (1, 1, 1, 1, 1, 2, 2, 2, 4, 4)


def client_rank(client_id: int, heterogeneous: bool) -> int:
    return HETEROGENEOUS_PATTERN[client_id % len(HETEROGENEOUS_PATTERN)] if heterogeneous else HOMOGENEOUS_RANK


def sample_clients(total_clients: int, clients_per_round: int, round_index: int, seed: int) -> list[int]:
    rng = np.random.default_rng([seed, 3_000_017, round_index])
    return sorted(int(i) for i in rng.choice(total_clients, size=clients_per_round, replace=False))


@dataclass(frozen=True)
class RoundResult:
    sampled: list[int]
    ranks: list[int]
    all_ranks: list[int]
    class_counts: np.ndarray  # total_clients x NUM_CLASSES
    sample_counts: list[int]  # of sampled clients
    weights: np.ndarray
    equal_weight_fallback: bool
    steps: FloristSteps
    payload: PayloadReport
    server: ServerCostEstimate
    notices: list[str]
    assumptions: list[str]


def simulate_round(
    total_clients: int,
    clients_per_round: int,
    round_index: int,
    alpha: float,
    heterogeneous: bool,
    tau: float,
    seed: int,
    bytes_per_param: int,
) -> RoundResult:
    if not (1 <= clients_per_round <= total_clients):
        raise ValueError("clients_per_round must be between 1 and total_clients")
    counts = dirichlet_label_counts(total_clients, NUM_CLASSES, SAMPLES_PER_CLASS, alpha, seed)
    sampled = sample_clients(total_clients, clients_per_round, round_index, seed)
    ranks = [client_rank(c, heterogeneous) for c in sampled]
    sizes = [int(counts[c].sum()) for c in sampled]

    notices: list[str] = []
    equal_fallback = sum(sizes) == 0
    if equal_fallback:
        weights = np.full(len(sampled), 1.0 / len(sampled))
        notices.append("No sampled client has any data this round, so equal weights are used.")
    else:
        weights = normalize_weights(sizes)
        empty = sum(1 for s in sizes if s == 0)
        if empty:
            notices.append(f"{empty} sampled client(s) have no data at this α, so their weight is 0.")

    Bs, As = synthetic_adapters(seed, sampled, ranks)
    steps = florist_steps(Bs, As, weights, tau)
    payload = payload_per_round(M, N, ranks, steps.p, bytes_per_param)
    server = estimate_server_ops(M, N, ranks, steps.p)

    if len(set(ranks)) > 1:
        notices.append("Client ranks differ, so FedIT is not computed (first release supports equal ranks only).")
    if not server.stacked_rank_below_min_dim:
        notices.append(
            f"Total stacked rank Σr_k = {server.stacked_rank} is at least min(m, n) = {min(M, N)}, "
            "so FLoRIST's compact-computation advantage no longer applies. Results stay correct."
        )

    return RoundResult(
        sampled=sampled,
        ranks=ranks,
        all_ranks=[client_rank(c, heterogeneous) for c in range(total_clients)],
        class_counts=counts,
        sample_counts=sizes,
        weights=weights,
        equal_weight_fallback=equal_fallback,
        steps=steps,
        payload=payload,
        server=server,
        notices=notices,
        assumptions=SYNTHETIC_ASSUMPTIONS
        + [
            f"Data: {NUM_CLASSES} classes x {SAMPLES_PER_CLASS} samples, split by Dirichlet(α) labels; clients are weighted by their sample counts.",
            "Clients are sampled without replacement, seeded by the random seed and the round number.",
        ],
    )
