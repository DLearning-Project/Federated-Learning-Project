"""Small numpy-only audio processing used by the acoustics explorer.

Everything here is deliberately dependency-light (numpy and the standard
library) so the web app runs without PyTorch. The log-Mel steps follow the
front end of the AudioSet-pretrained PANN models (Hann window, centered STFT,
power spectrogram, Slaney-scale Mel filters with area normalization, 10·log10
power, amin 1e-10). The real model computes these inside the network; here they
are reproduced only to *show* what the model receives. Check any numeric match
against the checkpoint's own front end before relying on it for training.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

# The matrices here are tiny (64 × 513 Mel weights × ~300 frames). OpenBLAS
# defaults to one thread per logical CPU, and on many-core laptops the thread
# coordination dominates: measured 174 ms per Mel product with 22 threads versus
# 0.7 ms with one. Limit BLAS to one thread for this process (the backend only
# does small calculations; offline training runs in its own process).
threadpool_limits(limits=1, user_api="blas")

MAX_READ_SECONDS = 3.0

# Slaney Mel scale (librosa's default): linear below 1 kHz, logarithmic above.
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOG_STEP = math.log(6.4) / 27.0


class UnsupportedAudio(ValueError):
    """The file is not a PCM WAV this module can read."""


def _pcm_to_float(raw: bytes, width: int) -> np.ndarray:
    """Convert little-endian integer PCM bytes to float64 in [-1, 1)."""
    if width == 1:  # 8-bit WAV is unsigned
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = (v ^ 0x800000) - 0x800000  # sign-extend 24-bit
        return v.astype(np.float64) / 8388608.0
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise UnsupportedAudio(f"unsupported sample width: {width} bytes")


def read_wav(path: Path, *, offset_s: float = 0.0, max_seconds: float = MAX_READ_SECONDS) -> tuple[np.ndarray, int]:
    """Read up to `max_seconds` of a PCM WAV starting at `offset_s`.

    Returns (mono float64 samples in [-1, 1), original sample rate). Only the
    requested window is read, so a 5-minute recording costs the same as a clip.
    """
    max_seconds = min(max_seconds, MAX_READ_SECONDS)
    try:
        with wave.open(str(path), "rb") as wf:
            rate, channels = wf.getframerate(), wf.getnchannels()
            width, total = wf.getsampwidth(), wf.getnframes()
            if total <= 0 or rate <= 0:
                raise UnsupportedAudio("empty WAV file")
            window = int(round(max_seconds * rate))
            # Shift the start earlier when the file is too short for a full window after `offset_s`.
            start = min(max(int(round(offset_s * rate)), 0), max(total - window, 0))
            count = min(window, total - start)
            wf.setpos(start)
            raw = wf.readframes(count)
    except (wave.Error, EOFError) as exc:
        raise UnsupportedAudio(f"cannot read WAV: {exc}") from exc

    samples = _pcm_to_float(raw, width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def fft_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling by truncating or zero-padding the spectrum.

    Exact for signals periodic over the clip; the clip edges can show tiny
    wrap-around artifacts, which is acceptable for a short display clip.
    """
    if sr_in == sr_out:
        return x.copy()
    n_in = len(x)
    n_out = int(round(n_in * sr_out / sr_in))
    spectrum = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=complex)
    keep = min(len(spectrum), len(out))
    out[:keep] = spectrum[:keep]
    return np.fft.irfft(out, n=n_out) * (n_out / n_in)


def hz_to_mel(hz) -> np.ndarray:
    f = np.asarray(hz, dtype=np.float64)
    mel = f / _F_SP
    return np.where(
        f >= _MIN_LOG_HZ,
        _MIN_LOG_MEL + np.log(np.maximum(f, 1e-10) / _MIN_LOG_HZ) / _LOG_STEP,
        mel,
    )


def mel_to_hz(mel) -> np.ndarray:
    m = np.asarray(mel, dtype=np.float64)
    hz = _F_SP * m
    return np.where(m >= _MIN_LOG_MEL, _MIN_LOG_HZ * np.exp(_LOG_STEP * (m - _MIN_LOG_MEL)), hz)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangular Mel filters with Slaney area normalization.

    Returns (weights of shape (n_mels, n_fft // 2 + 1), filter center frequencies in Hz).
    """
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    diffs = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / diffs[:-1, None]
    upper = ramps[2:] / diffs[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels]))[:, None]
    return weights, hz_points[1:-1]


def power_spectrogram(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """|STFT|² with a periodic Hann window and reflect padding (center=True).

    Returns shape (n_fft // 2 + 1, frames) with frames = 1 + len(x) // hop.
    """
    if len(x) <= n_fft // 2:
        raise ValueError("signal is shorter than half an FFT window")
    padded = np.pad(x, n_fft // 2, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    return (np.abs(np.fft.rfft(windows * hann, axis=1)) ** 2).T


def log_mel_db(power: np.ndarray, weights: np.ndarray, amin: float = 1e-10) -> np.ndarray:
    """Mel power → decibels (10·log10, reference 1.0, floor at `amin`)."""
    return 10.0 * np.log10(np.maximum(amin, weights @ power))


def synth_vessel(sr: int = 32000, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """A clearly artificial engine-plus-propeller signal for the no-data fallback.

    Engine harmonics of a 32 Hz fundamental up to ~700 Hz, a rising broadband
    "propeller" band between 1 and 8 kHz, and low background noise. Seeded, so
    every call returns the same signal. It is illustrative, not a recording.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    ramp = 0.35 + 0.65 * np.clip(t / seconds, 0.0, 1.0)  # boat approaches: louder over the clip

    engine = sum(np.sin(2 * np.pi * 32.0 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 23))
    engine *= 0.12 * ramp * (1.0 + 0.15 * np.sin(2 * np.pi * 2.0 * t))

    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spectrum *= ((freqs >= 1000.0) & (freqs <= 8000.0)) / np.sqrt(np.maximum(freqs, 1.0) / 1000.0)
    propeller = np.fft.irfft(spectrum, n=n)
    propeller *= 0.06 * ramp / max(float(np.std(propeller)), 1e-12)

    background = 0.004 * rng.standard_normal(n)
    signal = engine + propeller + background
    return signal / (2.0 * float(np.max(np.abs(signal))))


def rms_dbfs(x: np.ndarray) -> float:
    """Root-mean-square level in dB relative to full scale (1.0)."""
    return float(20.0 * np.log10(max(float(np.sqrt(np.mean(x**2))), 1e-12)))
