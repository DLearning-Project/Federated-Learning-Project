# Acoustics CNN training: plan, progress and run records

This is the single place that says what has been done for the acoustics
module, what comes next, and how every training run must be recorded so that
someone else can reproduce it. Setup commands live in [README.md](README.md).

**Goal.** Fine-tune an AudioSet-pretrained **PANN MobileNetV2** to detect
whether a vessel is present in a 3-second underwater recording (binary:
vessel / no vessel). Train it three ways on the same data and compare:

1. Centralized: one model sees all training data.
2. FedAvg with 3 simulated stations whose data is similar (IID).
3. FedAvg with 3 simulated stations whose data is skewed (non-IID).

The measured results will be shown on the acoustics page of the web app. The
trained model and preprocessing may later be reused by the KRAKEN buoy project,
which does not use federated learning.

**No model has been trained yet.** Nothing in this repository is a training
result. Every number on the web app's acoustics page is either a toy FedAvg
calculation or a spectrogram computed from audio (see "Done so far").

## Status at a glance

| Phase | What | Status |
| --- | --- | --- |
| 0 | Inspect repository, GPU and model/dataset sources | Done (2026-10-28) |
| 1 | Reproducible Python environment with GPU PyTorch | Done (2026-10-28, commit `b035139`) |
| — | Web app: spectrogram explorer (backend + frontend) | Done (backend `cda9c87`, frontend `cc50fc3`) |
| 2 | Dataset audit and leakage-safe splits | **Next.** Full dataset download in progress |
| 3 | Centralized baseline | Not started |
| 4 | FedAvg, IID and non-IID | Not started |
| 5 | Show measured results on the web app | Not started |
| 6 | Package the model for KRAKEN | Not started |

## Done so far

### Phase 0: inspection (2026-10-28)

Read the existing web app, confirmed the team conventions (`CalcEnvelope`
responses with `mode`, `units` and `assumptions`; one folder per module) and
checked the machine:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, driver 581.32
  (supports CUDA up to 13.0).
- Python 3.12 and 3.14 installed; 3.12 chosen.

### Phase 1: environment (2026-10-28, commit `b035139`)

Created `training/acoustics/.venv` (Python 3.12.10) with exact pins in
`requirements.txt` and a full freeze in `requirements.lock.txt`. Key choices:

- `torch==2.14.1+cu130`: newest build for CUDA 13.0, which the driver supports.
- No `torchaudio`: its CUDA builds stop at 2.11. Audio is read with
  `soundfile` and resampled with `soxr`.
- `torchlibrosa`: the STFT and log-Mel layers that PANN models run inside the
  network, so features are not computed twice.

`envinfo.py` records versions, GPU and Git state for run records, and smoke-tests
the GPU, including the PANN log-Mel front end. The verified report is
`results/env/2026-10-28_rtx4060-laptop.json`.

### Web app: spectrogram explorer (2026-10-29 to 2026-10-30)

Not training, but it uses the same preprocessing ideas, so it is listed here.

- **Backend** (`backend/app/modules/acoustics/`, commit `cda9c87`):
  `GET /api/v1/acoustics/clips` lists local QiandaoEar22 files.
  `POST /api/v1/acoustics/spectrogram` reads 3 s of audio, resamples it to
  32 kHz (or 16 kHz), and computes a log-Mel spectrogram with the PANN
  front-end settings, live on every request (about 30–40 ms). Real recordings
  are labeled `measured`; the built-in synthetic signal is labeled
  `illustrative`. 59 backend tests pass.
- **Frontend** (`frontend/src/modules/acoustics/`): the "What the model hears"
  card with a clip picker, six presets and Advanced controls.

The backend's resampler is a simple FFT method meant for display. Training
must use `soxr` (Phase 2/3) and record that choice.

## The training pipeline, step by step

Each phase ends with files that the next phase depends on. Do not skip ahead:
for example, a model trained before the splits are fixed cannot be compared
with later runs.

### Phase 2: dataset audit and splits

**Why this matters.** If 3-second clips cut from the same original recording
land in both the training and test sets, the model can recognize the
recording instead of the vessel, and the test score will be misleadingly high
("leakage"). Splits must therefore be made **by original recording**, before
anything else.

Steps:

1. Get the full dataset from an official source (see "Data source" below) and
   record the source URL, license, download date and a SHA-256 hash of every
   archive.
2. Write `audit.py` to inventory every file: path, label folder, sample rate,
   channels, bit depth, duration, whether it can be read, and the fields
   parsed from the file name.
3. Decide the binary labels: ship clips (`speedboat_target`, `speedboat_other`)
   = vessel present; background noise (`DATA0XXX`) = no vessel. Decide how to
   handle clips where all vessels are far and weak (`F_W`), and record the rule.
4. Confirm what identifies an original recording. The documentation describes
   the 14-digit field as the segment's start/stop time; the local files
   suggest clips from one recording share it (two local clips share
   `20220628101217` with different clip numbers). The documentation says there
   are 143 original ship recordings, which is a useful check.
5. Split by recording into train / validation / test (for example 70/15/15)
   with a fixed seed, balancing vessel and no-vessel counts where possible.
   Save `splits/split_manifest.csv` (one row per clip: path, label, group,
   split) and its SHA-256 hash.
6. Write down any limitation, for example if noise files have no recording
   group.

Outputs: `results/audit/` (inventory and summary), `splits/split_manifest.csv`.

### Phase 3: centralized baseline

**What fine-tuning means here.** The pretrained network already knows general
sound features from AudioSet. We keep those layers, replace the final layer
(the "head", which originally scored 527 AudioSet classes) with a new one that
outputs vessel / no vessel, and train on our data. Training only the head
first, then unfreezing more layers with a small learning rate, is a common way
to avoid destroying the pretrained features.

Steps:

1. Confirm the PANN MobileNetV2 implementation, checkpoint URL, license and
   input format (32 kHz mono waveform; the model computes the log-Mel itself).
   Record the checkpoint's SHA-256.
2. Write `dataset.py` (reads the split manifest, loads audio with `soundfile`,
   resamples with `soxr` to 32 kHz, returns waveform + label) and `model.py`
   (loads the checkpoint, replaces the head).
3. Check tensor shapes on a tiny batch.
4. **Overfit test:** train on about 20 clips until the model memorizes them. If
   it cannot, there is a bug; fix it before real training.
5. Train on the training split, checking the validation split after every
   epoch. Keep the checkpoint with the best validation score. Start with batch
   size 16 (8 GB GPU) and measure peak memory before increasing it.
6. Evaluate the chosen checkpoint on the **test split once**. Never tune
   settings based on test results.
7. Save the run record, metrics, confusion matrix, plots and checkpoint hash.

### Phase 4: FedAvg (IID and non-IID)

Simulate 3 stations on one GPU, one after another; no separate machines.

1. Split the **same training split** from Phase 2 among 3 clients: once at
   random (IID) and once skewed, for example by recording time or vessel type
   (non-IID). Save both partition manifests.
2. Each round: every client starts from the same global weights, trains
   locally for a few epochs, and sends back its weights. The server averages
   them **weighted by each client's number of training clips** (FedAvg).
3. Evaluate the global model on the fixed validation set every round; test once
   at the end, like Phase 3.
4. Keep the starting weights, preprocessing, test set and total compute
   comparable with Phase 3. Record rounds, local epochs, optimizer, wall time
   and estimated bytes sent.

### Phase 5: web app integration

Export measured metrics as JSON following the `CalcEnvelope` conventions
(`mode: "measured"`), and show training curves and client data summaries on
the acoustics page. Only show results for experiments that were actually run.

### Phase 6: KRAKEN reuse

Package the model and preprocessing with a documented input/output contract.
KRAKEN's hydrophone is believed to record at 48 kHz (to confirm); it would be
resampled to 32 kHz like the training data. Benchmark on a Raspberry Pi 4B and
evaluate on real lake recordings before claiming any performance.

## Data source

**What we have now.** 27 WAV files (330 MB) downloaded from a OneDrive link
into `training/acoustics/data/qiandaoear22/` (gitignored). Inspected
2026-10-30:

| Folder | Files | Format | Notes |
| --- | --- | --- | --- |
| `speedboat_target` | 10 | 52,734 Hz, mono, 32-bit, 3.0 s | One clip each from 10 timestamps |
| `speedboat_other` | 10 | 52,734 Hz, mono, 32-bit, 3.0 s | 6 are named "… - 副本" ("copy"). None matches another file among these 10, but their originals may be in the full folder: unconfirmed until the audit |
| `noise/data0001`, `noise/data0002` | 7 | 52,734 Hz, mono, 24-bit, 300.9 s | Full 5-minute recordings, not 3-second records |

**How this differs from the official documentation** (`QiandaoEar22_documentations.pdf`):

- The documentation says 16-bit samples; these files are 32-bit (ship clips)
  and 24-bit (noise).
- The documentation describes background noise as 25,900 three-second records;
  these noise files are 5-minute recordings.
- The documentation's file-name example includes a `DATA0008` field that these
  ship clip names do not have.
- The full dataset is about 209 archives of roughly 100 MB each, about
  **21 GB**, with 10,611 ship records and 25,900 noise records. 27 files are
  far too few to train or evaluate a CNN.

**Decision (2026-10-30).** Use the OneDrive files only for building and testing
code (the explorer, `audit.py`, `dataset.py`, the overfit test). Do **not**
train reported models or publish results from them. Before Phase 2 proper:

- ~~Find out who published the OneDrive link.~~ Answered (2026-11-01): it is
  the personal OneDrive of Xiaoyang Du (UCAS account `duxiaoyang22`), a
  citation author of the dataset (Shanghai Acoustics Laboratory, CAS). Record
  it as an author-hosted source, not an official repository.
- IEEE DataPort needs an account and often a subscription. Many universities
  provide institutional access: check with the library or the course
  instructor.
- Find the license. The documentation does not state one, and it decides
  whether excerpts or derived data may be committed to this repository.

**Hydrophone facts that affect modeling** (from the documentation):
DigitalHyd SR-1, 52,734 Hz, usable band **1–25.8 kHz**. Below 1 kHz the
response is outside the specified band, so low-frequency levels are not
calibrated. Resampling to 32 kHz also discards 16–25.8 kHz.

## Known issues to fix

- `backend/app/modules/acoustics/clips.py` skips every file named "副本" as a
  duplicate. In the 10-file sample none of them matches another file, so 6 of
  the 10 "other vessel" clips are hidden; whether that holds for the full
  folder is unconfirmed. Fix during Phase 2: skip a copy only if it is
  byte-identical to its original, and parse its name without the suffix so it
  keeps its recording timestamp.

## Run records (required for every training run)

Every run gets a unique run ID and a machine-readable record (YAML or JSON) in
`results/runs/<run_id>/`. Start from `envinfo.collect_env()` and add:

- **Who/why:** date and time, operator, experiment name, goal.
- **Code:** Git branch, commit SHA, dirty or clean.
- **Machine:** OS, Python and package versions, PyTorch, CUDA runtime and
  driver, GPU name and VRAM.
- **Data:** dataset source URL, version, license, archive hashes; label rule;
  class counts; split manifest path and hash; grouping rule; split sizes; seeds;
  IID/non-IID scheme and client counts.
- **Preprocessing:** decoding, channel handling, original and target sample
  rate, resampler, clip length, normalization, augmentation (training only).
- **Model:** implementation source and version, checkpoint URL, license and
  hash, number of outputs, frozen layers, head definition.
- **Training:** loss, optimizer, learning rate and schedule, batch size,
  epochs, early stopping, class weights, mixed precision, gradient clipping.
- **FL only:** rounds, local epochs, client participation, aggregation weights,
  optimizer resets, client sample counts, estimated bytes transferred.
- **Results:** per-epoch or per-round training and validation metrics;
  confusion matrix; per-class precision, recall and F1; macro F1; accuracy;
  training time; peak VRAM.
- **Selection:** the rule for picking the best checkpoint, its path and hash,
  and the single test evaluation.
- **Problems:** errors, deviations, anything unexpected.
- **Reproduce:** the exact commands.

Never edit a result file by hand, and never report a number that is not in a
saved record. Raw audio, `.venv/` and weight files stay out of Git.
