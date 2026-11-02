from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import CalcEnvelope, StrictModel


class LoraCalcInput(StrictModel):
    d_in: int = Field(..., ge=1, le=16384, description="Input dimension of the frozen weight matrix W")
    d_out: int = Field(..., ge=1, le=16384, description="Output dimension of W")
    rank: int = Field(..., ge=1, le=16384, description="LoRA rank limit r")
    clients: int = Field(..., ge=1, le=1000, description="Clients participating every round")
    rounds: int = Field(..., ge=1, le=1000, description="Communication rounds")
    bytes_per_param: Literal[1, 2, 4] = Field(..., description="1 = int8, 2 = fp16/bf16, 4 = fp32")

    @model_validator(mode="after")
    def rank_within_dims(self) -> "LoraCalcInput":
        limit = min(self.d_in, self.d_out)
        if self.rank > limit:
            raise ValueError(f"rank must be at most min(d_in, d_out) = {limit}")
        return self


class LoraCalcResult(BaseModel):
    full_params: int
    adapter_params: int
    a_params: int
    b_params: int
    adapter_fraction: float
    savings_ratio: float
    per_client_update_bytes: int
    per_round_upload_bytes: int
    total_upload_bytes: int
    total_download_bytes: int
    total_bytes: int
    full_update_bytes: int


LoraCalcResponse = CalcEnvelope[LoraCalcInput, LoraCalcResult]


# --- Simulator page ---------------------------------------------------------------


class SimulatorRoundInput(StrictModel):
    total_clients: int = Field(..., ge=2, le=50, description="Clients in the federation")
    clients_per_round: int = Field(..., ge=1, le=10, description="Clients sampled each round")
    round: int = Field(..., ge=1, le=100, description="Round number (1-based); changes the client sample")
    alpha: float = Field(..., ge=0.01, le=1000, description="Dirichlet concentration α for label partitioning")
    heterogeneous: bool = Field(..., description="Mixed client ranks (1, 2, 4) instead of rank 4 for all")
    tau: float = Field(..., gt=0, le=1, description="FLoRIST energy threshold τ")
    seed: int = Field(..., ge=0, le=1_000_000, description="Random seed")
    bytes_per_param: Literal[1, 2, 4] = Field(4, description="1 = int8, 2 = fp16/bf16, 4 = fp32")

    @model_validator(mode="after")
    def per_round_within_total(self) -> "SimulatorRoundInput":
        if self.clients_per_round > self.total_clients:
            raise ValueError(f"clients_per_round must be at most total_clients = {self.total_clients}")
        return self


class ErrorOut(BaseModel):
    absolute: float
    relative: float | None


class MethodPayloadOut(BaseModel):
    method: Literal["fedit", "exact_stack", "florist"]
    download_rank: int
    download_per_client_bytes: int
    download_total_bytes: int
    upload_total_bytes: int


class ServerCostOut(BaseModel):
    fedit_ops: int | None
    florist_ops: int
    direct_svd_ops: int
    stacked_rank: int
    stacked_rank_below_min_dim: bool
    assumptions: list[str]


class SimulatorRoundResult(BaseModel):
    provenance: Literal["computed_live_synthetic"] = "computed_live_synthetic"
    m: int
    n: int
    sampled_clients: list[int]
    client_ranks: list[int]
    all_client_ranks: list[int]
    sample_counts: list[int]
    weights: list[float]
    class_counts: list[list[int]]
    p: int
    singular_values: list[float]
    cumulative_energy: list[float]
    reconstruction_error: ErrorOut
    eq5_bound: float
    aggregation_error_fedit: ErrorOut | None
    aggregation_error_exact_stack: ErrorOut
    payload: list[MethodPayloadOut]
    payload_assumptions: list[str]
    server_cost: ServerCostOut
    notices: list[str]


class SimulatorRoundResponse(BaseModel):
    inputs: SimulatorRoundInput
    results: SimulatorRoundResult
    assumptions: list[str]


class TermLevels(BaseModel):
    beginner: str
    student: str
    researcher: str


class GlossaryTerm(BaseModel):
    id: str
    label: str
    short: str
    long: str
    equation: str
    equation_plain: str
    reference: str
    analogy: str | None = None
    levels: TermLevels
