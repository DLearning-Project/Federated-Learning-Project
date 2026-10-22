from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import CalcEnvelope, StrictModel

ParamValue = Annotated[float, Field(ge=-1e6, le=1e6)]


class ClientUpdate(StrictModel):
    name: str = Field(..., min_length=1, max_length=40)
    sample_count: int = Field(..., ge=0, le=10_000_000, description="Local training samples n_k")
    vector: list[ParamValue] = Field(..., min_length=1, max_length=8, description="Toy parameter vector")


class FedAvgInput(StrictModel):
    clients: list[ClientUpdate] = Field(..., min_length=2, max_length=10)

    @model_validator(mode="after")
    def check_clients(self) -> "FedAvgInput":
        if sum(c.sample_count for c in self.clients) <= 0:
            raise ValueError("total sample count must be greater than zero")
        lengths = {len(c.vector) for c in self.clients}
        if len(lengths) != 1:
            raise ValueError("all client vectors must have the same length")
        return self


class ClientWeight(BaseModel):
    name: str
    sample_count: int
    weight: float


class FedAvgResult(BaseModel):
    total_samples: int
    weights: list[ClientWeight]
    global_vector: list[float]
    unweighted_mean: list[float]


FedAvgResponse = CalcEnvelope[FedAvgInput, FedAvgResult]
