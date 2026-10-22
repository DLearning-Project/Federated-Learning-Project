"""Hand-checked calculations and validation errors for every module endpoint."""

import math

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
API = "/api/v1"


def test_health_and_modules():
    assert client.get(f"{API}/health").json()["status"] == "ok"
    ids = [m["id"] for m in client.get(f"{API}/modules").json()]
    assert ids == ["language-models", "acoustics", "hardware"]


# --- Language models -------------------------------------------------------

LORA = {"d_in": 4096, "d_out": 4096, "rank": 8, "clients": 10, "rounds": 50, "bytes_per_param": 2}


def test_lora_hand_calculation():
    body = client.post(f"{API}/language-models/calculate", json=LORA).json()
    r = body["results"]
    assert body["mode"] == "illustrative"
    assert r["full_params"] == 16_777_216  # 4096 * 4096
    assert r["adapter_params"] == 65_536  # 8 * (4096 + 4096)
    assert r["per_client_update_bytes"] == 131_072  # 65,536 * 2 bytes
    assert r["total_upload_bytes"] == 65_536_000  # 131,072 * 10 clients * 50 rounds
    assert r["total_download_bytes"] == r["total_upload_bytes"]
    assert math.isclose(r["savings_ratio"], 256.0)


def test_lora_rank_above_min_dim_rejected():
    res = client.post(f"{API}/language-models/calculate", json={**LORA, "d_in": 4, "rank": 8})
    assert res.status_code == 422
    assert "min(d_in, d_out) = 4" in res.json()["detail"]


def test_lora_bad_bytes_and_extra_field_rejected():
    assert client.post(f"{API}/language-models/calculate", json={**LORA, "bytes_per_param": 3}).status_code == 422
    assert client.post(f"{API}/language-models/calculate", json={**LORA, "extra": 1}).status_code == 422


# --- Acoustics -------------------------------------------------------------


def test_fedavg_hand_calculation():
    payload = {
        "clients": [
            {"name": "A", "sample_count": 100, "vector": [1.0, 0.0]},
            {"name": "B", "sample_count": 300, "vector": [0.0, 1.0]},
            {"name": "C", "sample_count": 0, "vector": [9.0, 9.0]},
        ]
    }
    r = client.post(f"{API}/acoustics/aggregate", json=payload).json()["results"]
    assert [w["weight"] for w in r["weights"]] == [0.25, 0.75, 0.0]
    assert r["global_vector"] == [0.25, 0.75]
    assert r["total_samples"] == 400


def test_fedavg_zero_total_rejected():
    payload = {"clients": [{"name": "A", "sample_count": 0, "vector": [1]}, {"name": "B", "sample_count": 0, "vector": [2]}]}
    res = client.post(f"{API}/acoustics/aggregate", json=payload)
    assert res.status_code == 422
    assert "greater than zero" in res.json()["detail"]


def test_fedavg_shape_mismatch_and_negative_rejected():
    mismatch = {"clients": [{"name": "A", "sample_count": 1, "vector": [1]}, {"name": "B", "sample_count": 1, "vector": [1, 2]}]}
    assert client.post(f"{API}/acoustics/aggregate", json=mismatch).status_code == 422
    negative = {"clients": [{"name": "A", "sample_count": -1, "vector": [1]}, {"name": "B", "sample_count": 2, "vector": [1]}]}
    assert client.post(f"{API}/acoustics/aggregate", json=negative).status_code == 422


# --- Hardware --------------------------------------------------------------


def test_profiles_listed():
    body = client.get(f"{API}/hardware/device-profiles").json()
    assert {"phone", "raspberry-pi", "jetson", "laptop"} <= {p["id"] for p in body["profiles"]}


def test_hardware_hand_calculation():
    # Phone: 60 s/epoch, 4 W training, 1.5 W radio.
    payload = {"profile_id": "phone", "update_mb": 10, "bandwidth_mbps": 20, "local_epochs": 3, "energy_budget_wh": 0.5}
    r = client.post(f"{API}/hardware/estimate", json=payload).json()["results"]
    assert math.isclose(r["upload_time_s"], 4.0)  # 10 MB * 8 / 20 Mbps
    assert math.isclose(r["communication_time_s"], 8.0)
    assert math.isclose(r["training_time_s"], 180.0)
    assert math.isclose(r["training_energy_wh"], 0.2)  # 180 s * 4 W / 3600
    assert math.isclose(r["communication_energy_wh"], 8 * 1.5 / 3600)
    assert r["meets_energy_budget"] is True
    assert r["time_budget_used"] is None


def test_hardware_invalid_inputs():
    base = {"profile_id": "phone", "update_mb": 10, "bandwidth_mbps": 20, "local_epochs": 3, "energy_budget_wh": 0.5}
    assert client.post(f"{API}/hardware/estimate", json={**base, "bandwidth_mbps": 0}).status_code == 422
    assert client.post(f"{API}/hardware/estimate", json={**base, "profile_id": "toaster"}).status_code == 404
