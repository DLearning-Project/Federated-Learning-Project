from fastapi import APIRouter

from . import service
from .schemas import LoraCalcInput, LoraCalcResponse

router = APIRouter(prefix="/language-models", tags=["language-models"])


@router.post("/calculate", response_model=LoraCalcResponse)
def calculate(body: LoraCalcInput) -> LoraCalcResponse:
    return service.calculate(body)
