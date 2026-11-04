# Acoustics training (offline)

Offline PyTorch experiments for the underwater acoustics module: binary
vessel-present / no-vessel detection with an AudioSet-pretrained PANN
MobileNetV2, first centralized, then FedAvg with 3 simulated clients (IID and
non-IID). Measured results will later be exported as JSON for
`backend/app/modules/acoustics/`. Nothing here runs inside the API.

Owner: Bryn. Status: **Phase 2 (dataset audit and splits) done.** No model has been trained
yet. For the full plan, progress so far, dataset notes and the required run
record fields, see [TRAINING.md](TRAINING.md).

## Layout

```text
training/acoustics/
  README.md               this file: setup and GPU verification
  TRAINING.md             plan, progress, dataset notes, run-record rules
  requirements.txt        direct dependencies, exact pins
  requirements.lock.txt   full `pip freeze` of the verified environment
  envinfo.py              environment provenance + GPU smoke test
  qiandao.py              what QiandaoEar22 file names and folders mean
  extract.py              hash the downloaded zips, extract data/qiandaoear22_full/
  audit.py                inventory, formats, duplicates -> results/audit/
  split.py                session-grouped train/val/test -> splits/
  tests/                  pytest tests (run: .venv/Scripts/python.exe -m pytest)
  results/env/            saved environment reports (committed)
  results/audit/          archive hashes, file inventory, audit summary (committed)
  splits/                 split manifest, its hash and summary (committed)
  data/                   raw audio and downloaded zips (gitignored)
  checkpoints/            model weights (gitignored)
  .venv/                  virtual environment (gitignored)
```

Later phases add `dataset.py`, `preprocessing.py`, `model.py`, `train.py`,
`evaluate.py` and `federated/`.

## 1. Set up the environment

Prerequisites: Windows 10/11 or Linux, **Python 3.12** and an NVIDIA GPU whose
driver supports **CUDA 13.0 or newer** (check the "CUDA Version" in the top
right of `nvidia-smi`). Python 3.14 is not used: several audio/ML packages lag
behind new Python releases.

From the repository root, in Git Bash:

```bash
cd training/acoustics
py -3.12 -m venv .venv                       # Linux/macOS: python3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
```

In PowerShell, use `.venv\Scripts\python.exe` instead. On Linux the
interpreter is `.venv/bin/python`. To activate the environment instead of typing
the full path: `source .venv/Scripts/activate` (Git Bash) or
`.venv\Scripts\Activate.ps1` (PowerShell).

`requirements.lock.txt` reproduces the exact verified environment.
`requirements.txt` lists only the direct dependencies. Use it when upgrading
deliberately, then regenerate the lockfile and save a new environment report.

### Why these versions

| Choice | Reason |
| --- | --- |
| Python 3.12 | Stable wheels for every package below; 3.14 is too new |
| `torch==2.14.1+cu130` | Newest build on the CUDA 13.0 index (2026-10-28). The 2.15 builds target CUDA 13.2, newer than the 581.32 driver supports |
| No `torchaudio` | Its CUDA wheels stop at 2.11 and do not match torch 2.14. Audio I/O uses `soundfile` and resampling uses `soxr` instead |
| `torchlibrosa` | The STFT and log-Mel layers that the original PANN models compute **inside** the network. Using them avoids computing Mel features twice |
| `librosa` | Required by `torchlibrosa` for Mel filter banks. Also used for audio inspection |

The PANN front-end settings (32 kHz, 1024-sample window, hop 320, 64 Mel bins,
50 Hz to 14 kHz) come from the original `audioset_tagging_cnn` configuration.
Check them against the checkpoint in Phase 3 before training.

## 2. Verify the GPU

```bash
nvidia-smi
.venv/Scripts/python.exe envinfo.py --save results/env/<date>_<machine>.json
```

`envinfo.py` prints and saves OS, Python and package versions, the CUDA runtime,
the cuDNN version, the driver version, GPU name and VRAM, and the Git branch,
commit and dirty status. It then runs a smoke test on the GPU:

1. a 1024×1024 matrix multiply, and
2. the PANN STFT → log-Mel front end on two 1-second random waveforms, checking
   that the output shape is `(2, 1, 101, 64)` and that every value is finite.

It exits with code 0 only if the smoke test passes. Use `--skip-smoke` to collect
versions only.

### Reference machine (verified 2026-10-28)

| Item | Value |
| --- | --- |
| OS | Windows 11 (10.0.26200) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, compute 8.9 |
| Driver / max CUDA | 581.32 / 13.0 |
| PyTorch / CUDA runtime / cuDNN | 2.14.1+cu130 / 13.0 / 9.24 |
| Python | 3.12.10 |

Full report: `results/env/2026-10-28_rtx4060-laptop.json`. With 8 GB of VRAM,
start Phase 3 with a small batch size (for example 16) and increase it only
after measuring peak memory.

## 3. Run records (applies to every later phase)

Every training or evaluation run gets a unique run ID and a machine-readable
record. Start each record from `envinfo.collect_env()` and add the fields
listed in [TRAINING.md](TRAINING.md#run-records-required-for-every-training-run).
Measured results go in `results/` and must never be hand-edited. Raw audio, `.venv/` and weight files (`*.pt`, `*.pth`, `*.ckpt`)
are gitignored.
