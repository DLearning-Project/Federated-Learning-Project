"""Illustrative per-round time and energy estimates for a federated client.

Formulas (one round):
    upload_time_s   = update_mb * 8 / bandwidth_mbps
    download_time_s = update_mb * 8 / bandwidth_mbps
    training_time_s = local_epochs * seconds_per_epoch
    energy_wh       = (training_time_s * train_power_w
                       + communication_time_s * radio_power_w) / 3600

Extension point (module owner): import real measurements (time, power, link
throughput) and calibrate profiles for a future digital twin. Keep that logic
behind this service so the router and UI do not change.
"""

import json
from functools import lru_cache
from pathlib import Path

from .schemas import (
    DeviceProfile,
    DeviceProfilesResponse,
    HardwareEstimateInput,
    HardwareEstimateResponse,
    HardwareEstimateResult,
)

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "device_profiles.json"

UNITS = {
    "upload_time_s": "seconds",
    "download_time_s": "seconds",
    "communication_time_s": "seconds",
    "training_time_s": "seconds",
    "total_time_s": "seconds",
    "training_energy_wh": "watt-hours",
    "communication_energy_wh": "watt-hours",
    "total_energy_wh": "watt-hours",
    "energy_budget_used": "fraction of energy budget",
    "time_budget_used": "fraction of time budget",
}

ASSUMPTIONS = [
    "Device profiles are illustrative, uncalibrated values, not real-device measurements.",
    "Bandwidth applies equally to upload and download; one upload and one download per round.",
    "1 MB = 8 megabits; no protocol overhead, retries, or latency.",
    "Training time scales linearly with local epochs.",
    "Communication energy (radio power × transfer time) is included in the total.",
    "Idle, screen, and cooling power are excluded.",
    "These estimates do not predict model accuracy.",
]


class UnknownProfileError(ValueError):
    pass


@lru_cache
def _load() -> DeviceProfilesResponse:
    raw = json.loads(DATA_FILE.read_text())
    return DeviceProfilesResponse(note=raw["note"], profiles=[DeviceProfile(**p) for p in raw["profiles"]])


def list_profiles() -> DeviceProfilesResponse:
    return _load()


def get_profile(profile_id: str) -> DeviceProfile:
    for p in _load().profiles:
        if p.id == profile_id:
            return p
    raise UnknownProfileError(f"unknown device profile '{profile_id}'")


def estimate(inp: HardwareEstimateInput) -> HardwareEstimateResponse:
    profile = get_profile(inp.profile_id)

    transfer_s = inp.update_mb * 8 / inp.bandwidth_mbps
    comm_s = 2 * transfer_s
    train_s = inp.local_epochs * profile.seconds_per_epoch
    train_wh = train_s * profile.train_power_w / 3600
    comm_wh = comm_s * profile.radio_power_w / 3600
    total_wh = train_wh + comm_wh
    total_s = train_s + comm_s

    time_used = total_s / inp.time_budget_s if inp.time_budget_s else None

    results = HardwareEstimateResult(
        profile_name=profile.name,
        upload_time_s=transfer_s,
        download_time_s=transfer_s,
        communication_time_s=comm_s,
        training_time_s=train_s,
        total_time_s=total_s,
        training_energy_wh=train_wh,
        communication_energy_wh=comm_wh,
        total_energy_wh=total_wh,
        energy_budget_used=total_wh / inp.energy_budget_wh,
        meets_energy_budget=total_wh <= inp.energy_budget_wh,
        time_budget_used=time_used,
        meets_time_budget=None if time_used is None else time_used <= 1,
    )
    return HardwareEstimateResponse(inputs=inp, results=results, units=UNITS, assumptions=ASSUMPTIONS)
