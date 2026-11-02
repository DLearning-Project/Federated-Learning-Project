"""HTTP-level checks for the spectrogram explorer endpoints.

Run with the rest of the backend tests (`pytest` from backend/, with
requirements-dev.txt installed). The audio comes from a tiny fake dataset built
in a temporary folder, so no real recordings are needed.
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_acoustics_dsp import SHIP_NAME, sine, write_wav

client = TestClient(app)
API = "/api/v1/acoustics"


class _FakeSettings:
    def __init__(self, path: Path):
        self.acoustics_data_path = path


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "qiandaoear22"
    write_wav(root / "speedboat_target" / SHIP_NAME, sine(1000, 52734, 3.0), 4, 52734)
    monkeypatch.setattr("app.modules.acoustics.router.get_settings", lambda: _FakeSettings(root))
    return root


def test_clips_lists_the_dataset(data_dir):
    body = client.get(f"{API}/clips").json()
    assert body["dataset_found"] is True
    assert [c["id"] for c in body["clips"]] == [f"speedboat_target/{SHIP_NAME}"]
    assert body["clips"][0]["recording_id"] == "20220628151815"


def test_clips_without_a_dataset_is_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr("app.modules.acoustics.router.get_settings", lambda: _FakeSettings(tmp_path / "missing"))
    res = client.get(f"{API}/clips")
    assert res.status_code == 200
    assert res.json()["dataset_found"] is False and res.json()["clips"] == []


def test_synthetic_spectrogram_works_without_data(tmp_path, monkeypatch):
    monkeypatch.setattr("app.modules.acoustics.router.get_settings", lambda: _FakeSettings(tmp_path / "missing"))
    body = client.post(f"{API}/spectrogram", json={}).json()
    assert body["mode"] == "illustrative"
    assert body["results"]["frames"] == 301 and body["results"]["mel_bins"] == 64
    assert np.array(body["results"]["spectrogram_db"]).shape == (64, 301)


def test_recording_spectrogram_is_measured(data_dir):
    payload = {"source": "recording", "clip_id": f"speedboat_target/{SHIP_NAME}", "n_mels": 32}
    body = client.post(f"{API}/spectrogram", json=payload).json()
    assert body["mode"] == "measured"
    assert body["results"]["mel_bins"] == 32
    assert body["inputs"]["n_mels"] == 32


def test_unknown_clip_is_404_with_a_message(data_dir):
    res = client.post(f"{API}/spectrogram", json={"source": "recording", "clip_id": "speedboat_target/nope.wav"})
    assert res.status_code == 404
    assert "clip" in res.json()["detail"].lower()


def test_path_traversal_is_refused(data_dir):
    res = client.post(f"{API}/spectrogram", json={"source": "recording", "clip_id": "../../etc/passwd"})
    assert res.status_code == 404


@pytest.mark.parametrize(
    "bad",
    [
        {"n_fft": 256, "hop_length": 512},
        {"sample_rate_hz": 16000, "fmax_hz": 14000},
        {"source": "recording"},
        {"n_mels": 500},
        {"unknown_field": 1},
    ],
)
def test_invalid_settings_return_422(data_dir, bad):
    res = client.post(f"{API}/spectrogram", json=bad)
    assert res.status_code == 422
    assert isinstance(res.json()["detail"], str)
