"""LoRA size and communication calculations.

Extension point (module owner): singular value thresholding and measured model
performance belong here behind new functions, not in the router or UI.
A future FLoRIST implementation must follow the official paper/code. Note that
averaging LoRA A and B independently is NOT equivalent to averaging their
products: mean(B_k) @ mean(A_k) != mean(B_k @ A_k) in general.
"""

from .schemas import LoraCalcInput, LoraCalcResponse, LoraCalcResult

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
