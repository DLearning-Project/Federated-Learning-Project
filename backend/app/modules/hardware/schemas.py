from pydantic import BaseModel, Field

from app.schemas.common import CalcEnvelope, StrictModel


class DeviceProfile(BaseModel):
    id: str
    name: str
    description: str
    seconds_per_epoch: float
    train_power_w: float
    radio_power_w: float
    typical_bandwidth_mbps: float


class DeviceProfilesResponse(BaseModel):
    mode: str = "illustrative"
    note: str
    profiles: list[DeviceProfile]


class HardwareEstimateInput(StrictModel):
    profile_id: str = Field(..., min_length=1, max_length=40)
    update_mb: float = Field(..., gt=0, le=10_000, description="Model update size per transfer, megabytes")
    bandwidth_mbps: float = Field(..., gt=0, le=10_000, description="Link bandwidth, megabits per second")
    local_epochs: int = Field(..., ge=1, le=100)
    energy_budget_wh: float = Field(..., gt=0, le=1_000, description="Per-round energy budget, watt-hours")
    time_budget_s: float | None = Field(None, gt=0, le=86_400, description="Optional per-round time budget, seconds")


class HardwareEstimateResult(BaseModel):
    profile_name: str
    upload_time_s: float
    download_time_s: float
    communication_time_s: float
    training_time_s: float
    total_time_s: float
    training_energy_wh: float
    communication_energy_wh: float
    total_energy_wh: float
    energy_budget_used: float
    meets_energy_budget: bool
    time_budget_used: float | None
    meets_time_budget: bool | None


HardwareEstimateResponse = CalcEnvelope[HardwareEstimateInput, HardwareEstimateResult]
