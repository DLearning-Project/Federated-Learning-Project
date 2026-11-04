"""Hand-checked audio math, clip catalog safety and the spectrogram service.

These tests need only numpy and pydantic (no FastAPI), so they run anywhere.
Tests that use real QiandaoEar22 audio are skipped when the data is not
downloaded; set ACOUSTICS_DATA_DIR to point at it if it lives elsewhere.
"""

import os
import wave
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.modules.acoustics import clips, dsp, service
from app.modules.acoustics.schemas import SpectrogramInput

SHIP_NAME = "20220628151815_M_SpeedBoat_N_S&SpeedBoat_M_M_1_label__1__9.wav"
SAME_COPY = "20220628151815_M_SpeedBoat_N_S&SpeedBoat_M_M_1_label__1__9 - 副本.wav"  # byte-identical to SHIP_NAME
DISTINCT_COPY = "20220628110727_M_Cargo_M_S&Cargo_M_S&QianDao_M_M_1_label__1__20 - 副本.wav"  # no original present


def write_wav(path: Path, samples: np.ndarray, width: int, rate: int, channels: int = 1) -> None:
    """Write float samples in [-1, 1) as little-endian PCM of `width` bytes."""
    scale = float(2 ** (8 * width - 1))
    ints = np.clip(np.round(samples * scale), -scale, scale - 1).astype("<i4")
    if width == 2:
        raw = ints.astype("<i2").tobytes()
    elif width == 3:
        raw = ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        raw = ints.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(raw)


def sine(freq: float, rate: int, seconds: float, amp: float = 0.5) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(int(rate * seconds)) / rate)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A tiny fake dataset with the same folder layout as the real one."""
    root = tmp_path / "qiandaoear22"
    write_wav(root / "speedboat_target" / SHIP_NAME, sine(1000, 52734, 3.0), 4, 52734)
    write_wav(root / "speedboat_target" / SAME_COPY, sine(1000, 52734, 3.0), 4, 52734)
    write_wav(root / "speedboat_other" / "20220628110227_M_Cargo_M_M&Cargo_F_W_1_label__1__21.wav", sine(300, 52734, 3.0), 4, 52734)
    write_wav(root / "speedboat_other" / DISTINCT_COPY, sine(700, 52734, 3.0), 4, 52734)
    # A 10 s background file is shorter than the 60 s offset: the last 3 s are used.
    write_wav(root / "noise" / "data0001" / "DATA0033.WAV", 0.01 * np.random.default_rng(1).standard_normal(8000 * 10), 3, 8000)
    return root


# --- Mel scale and filterbank ---------------------------------------------------


def test_slaney_mel_hand_values():
    assert dsp.hz_to_mel(1000.0) == pytest.approx(15.0)  # 1000 Hz / (200/3 Hz per mel)
    assert dsp.hz_to_mel(100.0) == pytest.approx(1.5)
    for hz in (50.0, 1000.0, 4000.0, 14000.0):
        assert dsp.mel_to_hz(dsp.hz_to_mel(hz)) == pytest.approx(hz)


def test_mel_filterbank_properties():
    weights, centers = dsp.mel_filterbank(32000, 1024, 64, 50.0, 14000.0)
    assert weights.shape == (64, 513)
    assert (weights >= 0).all()
    assert (weights.sum(axis=1) > 0).all()  # no empty filters at the PANN settings
    assert (np.diff(centers) > 0).all()
    assert centers[0] > 50.0 and centers[-1] < 14000.0
    # Slaney normalization: each triangle has area ~1 when it is wide enough to be well sampled.
    area = weights.sum(axis=1) * (32000 / 1024)
    assert area[20:] == pytest.approx(np.ones(44), rel=0.05)


# --- STFT -------------------------------------------------------------------------


def test_power_spectrogram_shape_and_peak_bin():
    x = sine(1000, 32000, 1.0, amp=0.5)  # 1000 Hz lands exactly on bin 32 of a 1024-point FFT
    power = dsp.power_spectrogram(x, n_fft=1024, hop=320)
    assert power.shape == (513, 1 + 32000 // 320)  # 101 frames, as in the PANN smoke test
    assert (power[:, 10:90].argmax(axis=0) == 32).all()
    # A Hann-windowed sine of amplitude A on an exact bin has |X| = A * N / 4.
    assert power[32, 50] == pytest.approx((0.5 * 1024 / 4) ** 2, rel=1e-6)


def test_tone_lands_in_the_right_mel_band():
    x = sine(1000, 32000, 1.0)
    weights, centers = dsp.mel_filterbank(32000, 1024, 64, 50.0, 14000.0)
    db = dsp.log_mel_db(dsp.power_spectrogram(x, 1024, 320), weights)
    band = int(db[:, 20:80].mean(axis=1).argmax())
    assert abs(centers[band] - 1000.0) < 60.0


# --- Reading and resampling ------------------------------------------------------


@pytest.mark.parametrize("width", [2, 3, 4])
def test_read_wav_matches_the_written_signal(tmp_path, width):
    ref = sine(440, 52734, 1.0, amp=0.5)
    write_wav(tmp_path / "t.wav", ref, width, 52734)
    x, rate = dsp.read_wav(tmp_path / "t.wav")
    assert rate == 52734
    assert x.shape == ref.shape
    assert np.abs(x - ref).max() < 1e-4


def test_read_wav_window_and_stereo_downmix(tmp_path):
    left = sine(440, 8000, 5.0, amp=0.4)
    right = -left
    write_wav(tmp_path / "s.wav", np.column_stack([left, right]).ravel(), 2, 8000, channels=2)
    x, rate = dsp.read_wav(tmp_path / "s.wav", offset_s=1.0, max_seconds=0.5)
    assert rate == 8000 and len(x) == 4000
    assert np.abs(x).max() < 1e-3  # opposite channels cancel in the mono downmix


def test_read_wav_rejects_non_wav(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not a wav file")
    with pytest.raises(dsp.UnsupportedAudio):
        dsp.read_wav(tmp_path / "bad.wav")


def test_fft_resample_keeps_the_tone():
    x = sine(1000, 52734, 3.0)  # 3000 whole cycles in the clip: periodic, so resampling is exact
    y = dsp.fft_resample(x, 52734, 32000)
    assert len(y) == 96000  # 158202 * 32000 / 52734
    assert np.abs(y - sine(1000, 32000, 3.0)).max() < 1e-9
    assert int(np.abs(np.fft.rfft(y)).argmax()) == 3000  # 1000 Hz at 1/3 Hz resolution


# --- Synthetic signal --------------------------------------------------------------


def test_synthetic_signal_is_deterministic_and_bounded():
    a, b = dsp.synth_vessel(), dsp.synth_vessel()
    assert np.array_equal(a, b)
    assert len(a) == 96000 and np.isfinite(a).all()
    assert np.abs(a).max() == pytest.approx(0.5)


def test_cached_synthetic_signal_is_read_only(tmp_path):
    service.spectrogram(SpectrogramInput(), tmp_path)
    cached = service._synthetic_samples()
    assert cached is service._synthetic_samples()
    assert not cached.flags.writeable  # a caller cannot corrupt the shared copy
    assert np.array_equal(cached, dsp.synth_vessel())


def test_blas_is_limited_to_one_thread():
    from threadpoolctl import threadpool_info

    blas = [p for p in threadpool_info() if p["user_api"] == "blas"]
    assert blas and all(p["num_threads"] == 1 for p in blas)


# --- Settings validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"n_fft": 256, "hop_length": 512},  # hop larger than the window
        {"sample_rate_hz": 16000, "fmax_hz": 14000.0},  # above Nyquist
        {"fmin_hz": 1000.0, "fmax_hz": 500.0},  # inverted band
        {"source": "recording"},  # recording without a clip id
        {"n_fft": 1000},  # not an allowed FFT size
        {"surprise": 1},  # unknown field
    ],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValidationError):
        SpectrogramInput(**bad)


# --- Clip catalog and path safety ----------------------------------------------------


def test_scan_and_labels(dataset):
    found = {c.category: c for c in clips.scan_clips(dataset)}
    assert set(found) == {"speedboat_target", "speedboat_other", "noise"}
    ship = found["speedboat_target"]
    assert ship.id == f"speedboat_target/{SHIP_NAME}"
    assert ship.label == "SpeedBoat (near, strong), SpeedBoat (medium, middle)"
    assert ship.recording_id == "20220628151815"
    assert found["noise"].recording_id == "DATA0033"
    assert len([c for c in clips.scan_clips(dataset) if c.category == "speedboat_target"]) == 1  # identical copy skipped


def test_distinct_copy_is_listed_under_its_recording(dataset):
    others = {c.id: c for c in clips.scan_clips(dataset) if c.category == "speedboat_other"}
    copy = others[f"speedboat_other/{DISTINCT_COPY}"]
    assert copy.recording_id == "20220628110727"
    assert copy.label == "Cargo (medium, strong), Cargo (medium, strong), QianDao (medium, middle)"
    assert clips.resolve_clip(dataset, copy.id).id == copy.id


def test_copy_that_differs_from_its_original_is_listed(dataset):
    write_wav(dataset / "speedboat_target" / SAME_COPY, sine(1500, 52734, 3.0), 4, 52734)  # now differs
    ids = {c.id for c in clips.scan_clips(dataset) if c.category == "speedboat_target"}
    assert ids == {f"speedboat_target/{SHIP_NAME}", f"speedboat_target/{SAME_COPY}"}


def test_scan_spreads_picks_over_many_files(tmp_path):
    root = tmp_path / "d"
    for i in range(40):
        write_wav(root / "speedboat_other" / f"20220628{i:06d}_M_Cargo_M_M_0_label__1__{i}.wav", sine(300, 8000, 0.2), 2, 8000)
    picked = clips.scan_clips(root, per_category=5)
    assert len(picked) == 5
    names = [c.id for c in picked]
    assert names[0].endswith("__0.wav") and names[-1].endswith("__39.wav")


@pytest.mark.parametrize(
    "bad_id",
    [
        "../secret.wav",
        "speedboat_target/../../../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "other_folder/file.wav",
        "speedboat_target/missing.wav",
        f"speedboat_target/{SAME_COPY}",
        "",
    ],
)
def test_resolve_clip_refuses_unsafe_or_unknown_ids(dataset, bad_id):
    with pytest.raises(clips.ClipNotFound):
        clips.resolve_clip(dataset, bad_id)


def test_resolve_clip_refuses_a_real_wav_outside_the_data_folder(dataset):
    outside = dataset.parent / "outside.wav"  # a genuine WAV, but not inside the dataset folder
    write_wav(outside, sine(440, 8000, 1.0), 2, 8000)
    assert outside.is_file()
    with pytest.raises(clips.ClipNotFound):
        clips.resolve_clip(dataset, "speedboat_target/../../outside.wav")


def test_resolve_clip_rejects_a_non_wav_inside_the_folder(dataset):
    (dataset / "speedboat_target" / "notes.txt").write_text("hi")
    with pytest.raises(clips.ClipNotFound):
        clips.resolve_clip(dataset, "speedboat_target/notes.txt")


def test_missing_dataset_folder(tmp_path):
    missing = tmp_path / "nope"
    listing = service.list_clips(missing)
    assert listing.dataset_found is False and listing.clips == []
    with pytest.raises(clips.ClipNotFound):
        clips.resolve_clip(missing, f"speedboat_target/{SHIP_NAME}")


# --- Service ---------------------------------------------------------------------------


def test_synthetic_spectrogram_defaults(tmp_path):
    out = service.spectrogram(SpectrogramInput(), tmp_path)  # no dataset needed
    r = out.results
    assert out.mode == "illustrative"
    assert (r.mel_bins, r.frames) == (64, 301)  # 3 s at 32 kHz with hop 320
    assert len(r.spectrogram_db) == 64 and len(r.spectrogram_db[0]) == 301
    assert r.empty_mel_filters == 0
    flat = np.array(r.spectrogram_db)
    assert np.isfinite(flat).all()
    assert flat.max() <= r.db_max + 0.05 and flat.min() >= r.db_min_shown - 0.05
    assert r.frame_seconds == pytest.approx(0.01)
    assert r.timings.total_ms >= 0


def test_coarse_settings_report_empty_filters(tmp_path):
    out = service.spectrogram(SpectrogramInput(n_fft=256, hop_length=256, n_mels=128), tmp_path)
    assert out.results.empty_mel_filters > 0  # too few FFT bins for that many low-frequency filters


def test_recording_spectrogram_is_measured(dataset):
    clip_id = f"speedboat_target/{SHIP_NAME}"
    out = service.spectrogram(SpectrogramInput(source="recording", clip_id=clip_id), dataset)
    r = out.results
    assert out.mode == "measured"
    assert r.original_sample_rate_hz == 52734 and r.clip_seconds == 3.0
    assert r.recording_id == "20220628151815"
    assert r.rms_dbfs == pytest.approx(-9.0, abs=0.1)  # sine of amplitude 0.5: 20·log10(0.5/√2)
    band = int(np.array(r.spectrogram_db)[:, 100:200].mean(axis=1).argmax())
    assert abs(r.mel_center_hz[band] - 1000.0) < 60.0  # the 1 kHz test tone shows up near 1 kHz


def test_short_background_file_uses_its_last_window(dataset):
    out = service.spectrogram(SpectrogramInput(source="recording", clip_id="noise/data0001/DATA0033.WAV"), dataset)
    assert out.results.clip_seconds == 3.0 and out.results.original_sample_rate_hz == 8000


def test_unknown_clip_raises(dataset):
    with pytest.raises(clips.ClipNotFound):
        service.spectrogram(SpectrogramInput(source="recording", clip_id="speedboat_target/nope.wav"), dataset)


# --- Real data (skipped when not downloaded) ----------------------------------------------


def _real_root() -> Path:
    env = os.environ.get("ACOUSTICS_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "training" / "acoustics" / "data" / "qiandaoear22"


@pytest.mark.skipif(not _real_root().is_dir(), reason="QiandaoEar22 audio is not downloaded")
def test_real_recordings_process_end_to_end():
    root = _real_root()
    found = clips.scan_clips(root, per_category=2)
    assert found, "data folder exists but holds no recognised clips"
    for clip in found:
        out = service.spectrogram(SpectrogramInput(source="recording", clip_id=clip.id), root)
        assert out.mode == "measured"
        assert out.results.frames == 301 and out.results.mel_bins == 64
        assert -90.0 < out.results.rms_dbfs < 0.0
        assert out.results.original_sample_rate_hz == 52734
