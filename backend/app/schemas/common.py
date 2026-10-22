"""Shared response conventions used by every module.

Every numerical endpoint returns a CalcEnvelope so the frontend can render
inputs, results, units and assumptions the same way for all modules.
"""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

InputT = TypeVar("InputT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)


class StrictModel(BaseModel):
    """Base for request models: reject unknown fields and non-finite floats."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CalcEnvelope(BaseModel, Generic[InputT, ResultT]):
    mode: Literal["illustrative"] = "illustrative"
    inputs: InputT
    results: ResultT
    units: dict[str, str]
    assumptions: list[str]


class ModuleInfo(BaseModel):
    id: str
    title: str
    description: str
    path: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
