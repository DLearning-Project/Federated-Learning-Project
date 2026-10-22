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
