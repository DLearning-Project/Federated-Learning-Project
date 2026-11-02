from typing import Any

from fastapi import APIRouter

from . import service
from .schemas import GlossaryTerm, LoraCalcInput, LoraCalcResponse, SimulatorRoundInput, SimulatorRoundResponse

router = APIRouter(prefix="/language-models", tags=["language-models"])


@router.post("/calculate", response_model=LoraCalcResponse)
def calculate(body: LoraCalcInput) -> LoraCalcResponse:
    return service.calculate(body)


@router.post("/simulator/round", response_model=SimulatorRoundResponse)
def simulator_round(body: SimulatorRoundInput) -> SimulatorRoundResponse:
    return service.simulator_round(body)


@router.get("/glossary", response_model=list[GlossaryTerm])
def glossary() -> list[GlossaryTerm]:
    return service.glossary()


@router.get("/paper")
def paper() -> dict[str, Any]:
    """Paper metadata and reported results (Tables 2-4), labeled paper-reported."""
    return service.paper()
