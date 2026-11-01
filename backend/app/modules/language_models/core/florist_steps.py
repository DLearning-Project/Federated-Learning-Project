"""Named intermediate states of one FLoRIST aggregation, for animation.

florist_steps() runs florist() once and returns every matrix the scenes
show, in pipeline order, each with its shape. It also reports two separate
error quantities:

- Aggregation error: a method's aggregate before compression versus the
  target delta_W = sum_k w_k B_k A_k. FedIT is generally nonzero; exact
  weighted stacking (FLoRIST's first step) is zero up to rounding.
- Reconstruction error: ||delta_W - B_g A_g||_F after thresholding, which
  equals the tail energy bound of Eq. 5.

delta_W itself is formed here only to measure errors on small teaching
examples; FLoRIST's computation never needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .aggregators import Matrices, fedit, florist, target_update


@dataclass(frozen=True)
class NamedState:
    name: str
    description: str
    value: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self.value.shape)


@dataclass(frozen=True)
class ErrorValue:
    absolute: float  # Frobenius norm
    relative: float | None  # absolute / ||delta_W||_F; None when delta_W is zero


@dataclass(frozen=True)
class FloristSteps:
    states: list[NamedState]  # pipeline order
    weights: np.ndarray
    ranks: list[int]
    p: int
    tau: float
    delta_w_shape: tuple[int, int]
    aggregation_error: dict[str, ErrorValue | None]  # "fedit" is None when ranks differ
    reconstruction_error: ErrorValue
    eq5_bound: float  # sqrt of discarded energy

    def state(self, name: str) -> NamedState:
        for s in self.states:
            if s.name == name:
                return s
        raise KeyError(name)


def _error(diff: np.ndarray, scale: float) -> ErrorValue:
    absolute = float(np.linalg.norm(diff, "fro"))
    return ErrorValue(absolute=absolute, relative=absolute / scale if scale > 0 else None)


def florist_steps(Bs: Matrices, As: Matrices, w: Sequence[float], tau: float) -> FloristSteps:
    res = florist(Bs, As, w, tau)
    weights = np.asarray(w, dtype=np.float64)
    ranks = [int(np.asarray(B).shape[1]) for B in Bs]

    states: list[NamedState] = []
    for k, (B, A) in enumerate(zip(Bs, As), start=1):
        states.append(NamedState(f"B_{k}", f"Client {k} adapter B (m x r_{k})", np.asarray(B, dtype=np.float64)))
        states.append(NamedState(f"A_{k}", f"Client {k} adapter A (r_{k} x n)", np.asarray(A, dtype=np.float64)))
    states += [
        NamedState("B_stack", "Clients' B side by side (Sec. 3)", res.B_stack),
        NamedState("A_stack", "Clients' weighted A stacked (Sec. 3)", res.A_stack),
        NamedState("U_B", "Left singular vectors of B_stack (Eq. 1)", res.U_B),
        NamedState("S_B", "Singular values of B_stack (Eq. 1)", res.S_B),
        NamedState("Vt_B", "Right singular vectors of B_stack, transposed (Eq. 1)", res.Vt_B),
        NamedState("U_A", "Left singular vectors of A_stack (Eq. 1)", res.U_A),
        NamedState("S_A", "Singular values of A_stack (Eq. 1)", res.S_A),
        NamedState("Vt_A", "Right singular vectors of A_stack, transposed (Eq. 1)", res.Vt_A),
        NamedState("Q", "Q = Vt_B U_A (Eq. 2)", res.Q),
        NamedState("P", "P = S_B Q S_A (Eq. 2)", res.P),
        NamedState("U_P", "Left singular vectors of P", res.U_P),
        NamedState("S_P", "Singular values of P = singular values of delta_W", res.singular_values),
        NamedState("Vt_P", "Right singular vectors of P, transposed", res.Vt_P),
        NamedState("cumulative_energy", "Share of energy kept by the first i components (Eq. 6)", res.cumulative_energy),
        NamedState("B_g", "Global adapter B_g = (U_B U_P)[:, :p] S_P[:p] (Eq. 3)", res.B_g),
        NamedState("A_g", "Global adapter A_g = (Vt_P Vt_A)[:p] (Eq. 3)", res.A_g),
    ]

    target = target_update(Bs, As, weights)
    scale = float(np.linalg.norm(target, "fro"))
    stacked = res.B_stack @ res.A_stack
    aggregation_error: dict[str, ErrorValue | None] = {
        "fedit": _error(fedit(Bs, As, weights) - target, scale) if len(set(ranks)) == 1 else None,
        "exact_stack": _error(stacked - target, scale),
    }
    reconstruction = _error(target - res.B_g @ res.A_g, scale)
    eq5_bound = float(np.sqrt(np.sum(res.singular_values[res.p:] ** 2)))

    return FloristSteps(
        states=states,
        weights=weights,
        ranks=ranks,
        p=res.p,
        tau=res.tau,
        delta_w_shape=(int(target.shape[0]), int(target.shape[1])),
        aggregation_error=aggregation_error,
        reconstruction_error=reconstruction,
        eq5_bound=eq5_bound,
    )


__all__ = ["ErrorValue", "FloristSteps", "NamedState", "florist_steps"]
