from fastapi import APIRouter, HTTPException

from . import service
from .schemas import DeviceProfilesResponse, HardwareEstimateInput, HardwareEstimateResponse

router = APIRouter(prefix="/hardware", tags=["hardware"])


@router.get("/device-profiles", response_model=DeviceProfilesResponse)
def device_profiles() -> DeviceProfilesResponse:
    return service.list_profiles()


@router.post("/estimate", response_model=HardwareEstimateResponse)
def estimate(body: HardwareEstimateInput) -> HardwareEstimateResponse:
    try:
        return service.estimate(body)
    except service.UnknownProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
