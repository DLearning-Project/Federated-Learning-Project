"""Environment provenance and GPU verification for acoustics training.

`collect_env()` returns the machine/software facts that every run record needs
(see training/acoustics/README.md). Run this file directly to verify the setup:

    .venv/Scripts/python.exe envinfo.py --save results/env/<name>.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Direct dependencies from requirements.txt whose versions go into run records.
PACKAGES = [
    "torch",
    "torchlibrosa",
    "librosa",
    "numpy",
    "scipy",
    "soundfile",
    "soxr",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "PyYAML",
    "tqdm",
    "pytest",
]

# PANN MobileNetV2 (AudioSet, 32 kHz) front-end settings from the original
# audioset_tagging_cnn config. Used only for the smoke test here.
PANN_SAMPLE_RATE = 32000
PANN_WINDOW_SIZE = 1024
PANN_HOP_SIZE = 320
PANN_MEL_BINS = 64
PANN_FMIN = 50
PANN_FMAX = 14000


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(repo_dir: Path) -> dict:
    status = _run(["git", "status", "--porcelain"], repo_dir)
    return {
        "remote": _run(["git", "config", "--get", "remote.origin.url"], repo_dir),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_dir),
        "commit": _run(["git", "rev-parse", "HEAD"], repo_dir),
        "dirty": None if status is None else bool(status),
    }


def gpu_info() -> dict:
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "devices": [],
        "driver": None,
    }
    if info["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_mib": props.total_memory // (1024 * 1024),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        info["driver"] = smi.splitlines()[0].strip()
    return info


def package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_env(repo_dir: Path | None = None) -> dict:
    repo_dir = repo_dir or Path(__file__).resolve().parent
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "gpu": gpu_info(),
        "git": git_info(repo_dir),
    }


def smoke_test() -> dict:
    """Run a matmul and the PANN log-Mel front end on the GPU."""
    import torch
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    device = torch.device("cuda")
    torch.manual_seed(0)

    a = torch.randn(1024, 1024, device=device)
    matmul_err = (a @ torch.eye(1024, device=device) - a).abs().max().item()

    spectrogram = Spectrogram(
        n_fft=PANN_WINDOW_SIZE, hop_length=PANN_HOP_SIZE, win_length=PANN_WINDOW_SIZE,
        window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
    ).to(device)
    logmel = LogmelFilterBank(
        sr=PANN_SAMPLE_RATE, n_fft=PANN_WINDOW_SIZE, n_mels=PANN_MEL_BINS,
        fmin=PANN_FMIN, fmax=PANN_FMAX, ref=1.0, amin=1e-10, top_db=None,
        freeze_parameters=True,
    ).to(device)

    # Two 1-second clips of (batch, samples) waveform, as PANN models expect.
    waveform = torch.randn(2, PANN_SAMPLE_RATE, device=device) * 0.1
    features = logmel(spectrogram(waveform))  # (batch, 1, frames, mel_bins)
    torch.cuda.synchronize()

    expected = (2, 1, PANN_SAMPLE_RATE // PANN_HOP_SIZE + 1, PANN_MEL_BINS)
    if tuple(features.shape) != expected:
        raise RuntimeError(f"log-Mel shape {tuple(features.shape)} != expected {expected}")
    if not torch.isfinite(features).all():
        raise RuntimeError("log-Mel output contains non-finite values")

    return {
        "matmul_max_abs_error": matmul_err,
        "logmel_shape": list(features.shape),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", type=Path, help="write the report as JSON to this path")
    parser.add_argument("--skip-smoke", action="store_true", help="only collect versions")
    args = parser.parse_args()

    report = collect_env()
    ok = True
    if not args.skip_smoke:
        try:
            report["smoke_test"] = {"passed": True, **smoke_test()}
        except Exception as exc:  # report the failure instead of a traceback
            report["smoke_test"] = {"passed": False, "error": repr(exc)}
            ok = False

    text = json.dumps(report, indent=2)
    print(text)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
