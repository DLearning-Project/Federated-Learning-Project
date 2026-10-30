"""Acoustics module: toy FedAvg aggregation and the spectrogram explorer.

`aggregate` is a toy calculation on small vectors. `spectrogram` processes real
audio (or a clearly labeled synthetic signal) on request; it is cheap enough to
run per slider change. Real model training must run offline in a separate
worker (see training/acoustics/), never inside an API request, and must keep
evaluation data separate from client training data.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import clips, dsp
from .schemas import (
    ClientWeight,
    ClipInfo,
    ClipList,
    FedAvgInput,
    FedAvgResponse,
    FedAvgResult,
    SpectrogramInput,
    SpectrogramResponse,
    SpectrogramResult,
    StageTimings,
)

UNITS = {
    "total_samples": "samples",
    "weight": "fraction of total samples (sums to 1)",
    "global_vector": "same units as the client parameter vectors",
    "unweighted_mean": "same units as the client parameter vectors",
}

ASSUMPTIONS = [
    "Vectors are small toy parameters, not real PANN MobileNetV2 weights.",
    "Each client's weight is n_k / Σ n: its share of all training samples.",
    "Global vector = Σ_k weight_k · vector_k (one FedAvg aggregation step).",
    "Clients with zero samples receive zero weight.",
    "No CNN is trained and no classification accuracy is predicted.",
]

FOLDER_HINT = "training/acoustics/data/qiandaoear22"
DISPLAY_RANGE_DB = 80.0  # shown dynamic range below the loudest cell
SYNTH_RATE_HZ = 32000

SPECTROGRAM_UNITS = {
    "spectrogram_db": "dB re 1.0 full-scale power (10·log10 of Mel power)",
    "mel_center_hz": "Hz",
    "rms_dbfs": "dB relative to full scale",
    "frame_seconds": "seconds between frames",
    "timings": "milliseconds, measured on the server for this request",
}


def aggregate(inp: FedAvgInput) -> FedAvgResponse:
    total = sum(c.sample_count for c in inp.clients)
    dim = len(inp.clients[0].vector)
    weights = [c.sample_count / total for c in inp.clients]

    global_vector = [
        sum(w * c.vector[i] for w, c in zip(weights, inp.clients)) for i in range(dim)
    ]
    unweighted = [sum(c.vector[i] for c in inp.clients) / len(inp.clients) for i in range(dim)]

    results = FedAvgResult(
        total_samples=total,
        weights=[
            ClientWeight(name=c.name, sample_count=c.sample_count, weight=w)
            for c, w in zip(inp.clients, weights)
        ],
        global_vector=global_vector,
        unweighted_mean=unweighted,
    )
    return FedAvgResponse(inputs=inp, results=results, units=UNITS, assumptions=ASSUMPTIONS)


def list_clips(data_dir: Path) -> ClipList:
    found = clips.scan_clips(data_dir)
    return ClipList(
        dataset_found=data_dir.is_dir(),
        folder_hint=FOLDER_HINT,
        clips=[ClipInfo(id=c.id, label=c.label, category=c.category, recording_id=c.recording_id) for c in found],
    )


@lru_cache(maxsize=1)
def _synthetic_samples() -> np.ndarray:
    """The seeded synthetic signal never changes, so build it once (~95 ms) and share it read-only."""
    samples = dsp.synth_vessel(SYNTH_RATE_HZ)
    samples.flags.writeable = False
    return samples


def spectrogram(inp: SpectrogramInput, data_dir: Path) -> SpectrogramResponse:
    """Load audio, resample, and compute a log-Mel spectrogram for display.

    Raises clips.ClipNotFound for an unknown recording and dsp.UnsupportedAudio
    for a file that is not a PCM WAV.
    """
    t0 = time.perf_counter()
    if inp.source == "recording":
        clip = clips.resolve_clip(data_dir, inp.clip_id or "")
        samples, original_rate = dsp.read_wav(clip.path, offset_s=clip.offset_s)
        label, recording_id = clip.label, clip.recording_id
    else:
        original_rate = SYNTH_RATE_HZ
        samples = _synthetic_samples()
        label, recording_id = "Synthetic engine and propeller signal (not a recording)", None
    t1 = time.perf_counter()

    audio = dsp.fft_resample(samples, original_rate, inp.sample_rate_hz)
    if len(audio) <= inp.n_fft // 2:
        raise dsp.UnsupportedAudio("The clip is too short for this FFT size.")
    t2 = time.perf_counter()

    power = dsp.power_spectrogram(audio, inp.n_fft, inp.hop_length)
    t3 = time.perf_counter()

    weights, centers = dsp.mel_filterbank(inp.sample_rate_hz, inp.n_fft, inp.n_mels, inp.fmin_hz, inp.fmax_hz)
    db = dsp.log_mel_db(power, weights)
    t4 = time.perf_counter()

    db_max = float(db.max())
    shown = np.clip(db, db_max - DISPLAY_RANGE_DB, db_max)
    ms = lambda a, b: round((b - a) * 1000.0, 2)  # noqa: E731
    results = SpectrogramResult(
        source_label=label,
        recording_id=recording_id,
        original_sample_rate_hz=original_rate,
        clip_seconds=round(len(samples) / original_rate, 3),
        rms_dbfs=round(dsp.rms_dbfs(samples), 1),
        frames=int(db.shape[1]),
        mel_bins=int(db.shape[0]),
        frame_seconds=round(inp.hop_length / inp.sample_rate_hz, 5),
        empty_mel_filters=int((weights.sum(axis=1) == 0).sum()),
        db_max=round(db_max, 1),
        db_min_shown=round(db_max - DISPLAY_RANGE_DB, 1),
        mel_center_hz=[round(float(f), 1) for f in centers],
        spectrogram_db=np.round(shown, 1).tolist(),
        timings=StageTimings(
            read_ms=ms(t0, t1), resample_ms=ms(t1, t2), stft_ms=ms(t2, t3), mel_ms=ms(t3, t4), total_ms=ms(t0, t4)
        ),
    )
    assumptions = [
        "Mel filters, window and dB scaling follow the AudioSet-pretrained PANN front end "
        "(Slaney Mel scale, Hann window, power spectrogram, 10·log10, amin 1e-10).",
        f"Display only: values are clipped to the loudest cell minus {DISPLAY_RANGE_DB:.0f} dB. "
        "The model itself does not clip.",
        "Audio is resampled by a band-limited FFT method; clip edges may show tiny artifacts.",
        "The window shown is the first 3 s of a clip, or 3 s starting at 60 s of a background recording.",
    ]
    if inp.source == "synthetic":
        assumptions.insert(0, "Synthetic signal generated by the server: illustrative, not measured.")
    else:
        assumptions.insert(0, "Computed from a real QiandaoEar22 recording: measured input, no model involved.")
    return SpectrogramResponse(
        mode="measured" if inp.source == "recording" else "illustrative",
        inputs=inp,
        results=results,
        units=SPECTROGRAM_UNITS,
        assumptions=assumptions,
    )
