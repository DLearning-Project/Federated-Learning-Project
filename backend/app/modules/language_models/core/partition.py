"""Dirichlet(alpha) label partitioning across clients.

For every class, the class's samples are split across clients in proportions
drawn from Dir(alpha, ..., alpha). Small alpha gives each client only a few
dominant classes (highly non-IID); large alpha approaches an even split. This
follows the per-category scheme of the official repo's
client_data_allocation.py (Dolly), without its minimum-size resampling:
clients may receive zero samples, and callers must handle that.
"""

from __future__ import annotations

import numpy as np


def dirichlet_label_counts(
    num_clients: int, num_classes: int, samples_per_class: int, alpha: float, seed: int
) -> np.ndarray:
    """Return an int array (num_clients x num_classes) of sample counts.

    Each column sums to samples_per_class. Same arguments give the same result.
    """
    if num_clients < 1 or num_classes < 1 or samples_per_class < 0:
        raise ValueError("num_clients and num_classes must be positive; samples_per_class non-negative")
    if not (alpha > 0 and np.isfinite(alpha)):
        raise ValueError("alpha must be a positive finite number")
    rng = np.random.default_rng([seed, 7919])
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for c in range(num_classes):
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts[:, c] = rng.multinomial(samples_per_class, proportions)
    return counts
