"""LoRA size and communication calculations.

Extension point (module owner): singular value thresholding and measured model
performance belong here behind new functions, not in the router or UI.
A future FLoRIST implementation must follow the official paper/code. Note that
averaging LoRA A and B independently is NOT equivalent to averaging their
products: mean(B_k) @ mean(A_k) != mean(B_k @ A_k) in general.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .core.simulator import simulate_round
from .core.synthetic import M, N
from .schemas import (
    ErrorOut,
    GlossaryTerm,
    LoraCalcInput,
    LoraCalcResponse,
    LoraCalcResult,
    MethodPayloadOut,
    ServerCostOut,
    SimulatorRoundInput,
    SimulatorRoundResponse,
    SimulatorRoundResult,
)

PAPER_FACTS_PATH = Path(__file__).resolve().parent / "data" / "paper_facts.json"

UNITS = {
    "full_params": "parameters",
    "adapter_params": "parameters",
    "a_params": "parameters",
    "b_params": "parameters",
    "adapter_fraction": "ratio (adapter / full)",
    "savings_ratio": "× fewer parameters",
    "per_client_update_bytes": "bytes",
    "per_round_upload_bytes": "bytes",
    "total_upload_bytes": "bytes",
    "total_download_bytes": "bytes",
    "total_bytes": "bytes",
    "full_update_bytes": "bytes",
}

ASSUMPTIONS = [
    "One weight matrix W (d_out × d_in) is adapted with A (r × d_in) and B (d_out × r).",
    "The base model stays frozen; only A and B are trained and exchanged.",
    "Upload and download carry the same adapter size.",
    "All selected clients participate in every round.",
    "Initial distribution of the base model to clients is excluded.",
    "Rank r is a limit: the product B·A has rank at most r, and may be lower in practice.",
    "No compression, encryption, or protocol overhead is counted.",
]


def calculate(inp: LoraCalcInput) -> LoraCalcResponse:
    full = inp.d_in * inp.d_out
    a_params = inp.rank * inp.d_in
    b_params = inp.d_out * inp.rank
    adapter = a_params + b_params
    update_bytes = adapter * inp.bytes_per_param
    per_round = update_bytes * inp.clients
    upload = per_round * inp.rounds
    # Each participating client downloads the aggregated adapter every round.
    download = upload

    results = LoraCalcResult(
        full_params=full,
        adapter_params=adapter,
        a_params=a_params,
        b_params=b_params,
        adapter_fraction=adapter / full,
        savings_ratio=full / adapter,
        per_client_update_bytes=update_bytes,
        per_round_upload_bytes=per_round,
        total_upload_bytes=upload,
        total_download_bytes=download,
        total_bytes=upload + download,
        full_update_bytes=full * inp.bytes_per_param,
    )
    return LoraCalcResponse(inputs=inp, results=results, units=UNITS, assumptions=ASSUMPTIONS)


# --- Simulator page ---------------------------------------------------------------


def _err(e) -> ErrorOut:
    return ErrorOut(absolute=e.absolute, relative=e.relative)


def simulator_round(inp: SimulatorRoundInput) -> SimulatorRoundResponse:
    r = simulate_round(
        total_clients=inp.total_clients,
        clients_per_round=inp.clients_per_round,
        round_index=inp.round,
        alpha=inp.alpha,
        heterogeneous=inp.heterogeneous,
        tau=inp.tau,
        seed=inp.seed,
        bytes_per_param=inp.bytes_per_param,
    )
    s = r.steps
    fedit_err = s.aggregation_error["fedit"]
    results = SimulatorRoundResult(
        m=M,
        n=N,
        sampled_clients=r.sampled,
        client_ranks=r.ranks,
        all_client_ranks=r.all_ranks,
        sample_counts=r.sample_counts,
        weights=[float(w) for w in r.weights],
        class_counts=r.class_counts.tolist(),
        p=s.p,
        singular_values=[float(v) for v in s.state("S_P").value],
        cumulative_energy=[float(v) for v in s.state("cumulative_energy").value],
        reconstruction_error=_err(s.reconstruction_error),
        eq5_bound=s.eq5_bound,
        aggregation_error_fedit=_err(fedit_err) if fedit_err else None,
        aggregation_error_exact_stack=_err(s.aggregation_error["exact_stack"]),
        payload=[
            MethodPayloadOut(
                method=mp.method,
                download_rank=mp.download_rank,
                download_per_client_bytes=mp.download_per_client_bytes,
                download_total_bytes=mp.download_total_bytes,
                upload_total_bytes=mp.upload_total_bytes,
            )
            for mp in r.payload.methods.values()
        ],
        payload_assumptions=r.payload.assumptions,
        server_cost=ServerCostOut(
            fedit_ops=r.server.fedit_ops,
            florist_ops=r.server.florist_ops,
            direct_svd_ops=r.server.direct_svd_ops,
            stacked_rank=r.server.stacked_rank,
            stacked_rank_below_min_dim=r.server.stacked_rank_below_min_dim,
            assumptions=r.server.assumptions,
        ),
        notices=r.notices,
    )
    return SimulatorRoundResponse(inputs=inp, results=results, assumptions=r.assumptions)


@lru_cache(maxsize=1)
def _paper_facts() -> dict[str, Any]:
    return json.loads(PAPER_FACTS_PATH.read_text(encoding="utf-8"))


def glossary() -> list[GlossaryTerm]:
    return [GlossaryTerm(**t) for t in _paper_facts()["terms"]]


def paper() -> dict[str, Any]:
    facts = _paper_facts()
    return {"provenance": "paper_reported", "paper": facts["paper"], "reported_results": facts["reported_results"]}
