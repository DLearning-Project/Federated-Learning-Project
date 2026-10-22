from fastapi import APIRouter

from . import service
from .schemas import FedAvgInput, FedAvgResponse

router = APIRouter(prefix="/acoustics", tags=["acoustics"])


@router.post("/aggregate", response_model=FedAvgResponse)
def aggregate(body: FedAvgInput) -> FedAvgResponse:
    return service.aggregate(body)
