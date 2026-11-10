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
| 2 | Dataset audit and leakage-safe splits | Done (2026-11-03; split balanced by distance 2026-11-09) |
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

### Phase 2: audit and splits (2026-11-03)

Code: `qiandao.py` (what names and folders mean), `extract.py`, `audit.py`,
`split.py`, tests in `tests/test_phase2.py` (21 pass). Reproduce from
`training/acoustics/`:

```bash
.venv/Scripts/python.exe extract.py   # hash zips, extract to data/qiandaoear22_full/
.venv/Scripts/python.exe audit.py     # results/audit/inventory.csv, summary.json
.venv/Scripts/python.exe split.py     # splits/split_manifest.csv (+ .sha256), split_summary.json
```

**Audit findings** (`results/audit/summary.json`):

- 26,338 files in 17 archives, all readable. Only **10,637 are distinct**
  (by SHA-256): 10,587 ship clips (8.8 h, all 52,734 Hz mono 32-bit 3.0 s)
  and 50 background files (4.2 h, 24-bit, 300.94 s each). The documentation
  says 10,611 ship records from 143 recordings; we find 130 recordings.
- Each folder is one "target vs. other" task built from the same pool, so
  the same clip is stored up to four times. UUV/* and noise_target/target add
  no new audio; SpeedBoat and KaiYuan hold every distinct ship clip.
- All 2,578 "… - 副本" files are byte-identical to a clip with the original
  name in another folder. They add nothing.
- DATA0145–0147 exist on both cards (DATA0001, DATA0002) with different
  content: card + number identifies a background file, not the number alone.
- Recordings are 5 minutes long and start 300–302 s apart, forming **28
  continuous sessions** (the largest holds 2,110 clips). The `label__k`
  field is just the single/multiple flag, not a class.
- Name quirks kept as they are, but counted: sources without distance or
  audibility (`ArtificialSignals` 972, `Unknown` 46, `No7` 28,
  `QianDao_M_MS` 10, `No7_Slow_Down` 8), 51 empty source names, and
  `KaiYan` (191 clips, probably a typo of `KaiYuan`).

**Decisions (2026-11-03):**

- Labels: vessel = any ship clip; no vessel = 3-second windows (offsets
  0, 3, … 297 s) of the background files.
- Clips where every vessel is far and weak (1,171) are **kept** as vessel,
  because a deployed buoy mostly hears distant vessels. They are flagged
  `all_far_weak`, and Phase 3 must report their recall separately.
- The 100 clips whose only source is `Car` (far, weak) are **excluded**:
  not a watercraft and not clean background. Clips that mix a vessel with
  `ArtificialSignals` or `Helicopter` stay.
- Grouping is by **session**, not single recording: neighbouring recordings
  hear the same vessel pass, so a recording-level split would still leak.
  Noise sessions are runs of consecutive file numbers on one card (6 runs).
- **Revised 2026-11-09: 30-minute blocks with gaps.** Splitting whole
  sessions left validation at 19% and test at 23% near-vessel clips against
  45% in train, because two sessions hold 71% of all near clips and each is
  larger than a whole 15% split. Ship sessions longer than 6 recordings are
  now cut into blocks of 6 (30 min); the 5-minute recording between two
  blocks is a gap that joins them if both land in the same split and is
  dropped otherwise. No two splits hold recordings less than 5 minutes apart
  (`split.py` checks this), at the cost of 4 recordings (395 clips, 3.8%).
- The assignment minimizes size mismatch plus the mismatch in
  closest-vessel distance mix (near/medium/far), best of 300 seeded restarts
  improved by moving or swapping blocks (seed `20261103`). `split.py`
  refuses any split where a group, recording or file hash appears in two
  splits.
- Why not a perfectly "real-world" split: a deployed buoy hears unseen time
  with whatever vessels pass, which whole sessions imitate best. But the
  project compares centralized and FedAvg training on one test set, so a
  test set skewed by chance would make that comparison noisy. Realism comes
  from the evaluation instead (see Phase 3).

**Result** (`splits/split_summary.json`, manifest SHA-256 `b6a43609…5c47`):

| Split | Vessel | No vessel | Ship blocks + noise sessions | Near / medium / far | All far-and-weak |
| --- | --- | --- | --- | --- | --- |
| train | 7,208 (71.4%) | 3,600 (72%) | 28 + 2 | 37 / 51 / 12% | 802 |
| val | 1,439 (14.3%) | 700 (14%) | 7 + 1 | 36 / 53 / 10% | 23 |
| test | 1,445 (14.3%) | 700 (14%) | 6 + 3 | 36 / 53 / 11% | 158 |

**Limitations to keep in mind:**

- Splits still differ in some content: test has no 2022-06-26 clips,
  validation has only 23 all-far-and-weak clips (report far-and-weak recall
  on test), and validation noise comes from one session.
- A vessel heard at the end of one block could in principle still be heard
  after the 5-minute gap; this is a much smaller risk than splitting by clip,
  but not zero.
- Background files have no date or link to the ship recordings, so the model
  could partly learn "which card or day" instead of "vessel or not". Look for
  this in Phase 3 errors.
- Vessel examples outnumber no-vessel windows about 2 : 1; Phase 3 must
  account for it (class weights or balanced sampling) and report per-class
  metrics.
- The background clips were hand-picked "without targets and other
  interference" (paper, §2.2), so they are cleaner than what a buoy hears;
  false-alarm rates measured on them are optimistic.
- The audit summary records `dirty: true` only because an unrelated,
  untracked report draft (`docs/`) was present; the code was commit
  `f09bab8` with no modified tracked files.

**Comparison with the authors' results.** The paper (Du & Hong,
arXiv:2406.04354, §3.1) reports 99.79% accuracy (CRNN, log-Mel) on the
same binary task, using "the first 5000 ship noise recordings" and the same
50 background files, split 70/15/15 with no mention of grouping by recording.
Neighbouring 3-second clips of one pass were then probably in both training
and test, so that figure is likely optimistic. Expect lower numbers from our
leakage-safe split; that difference is a finding, not a failure.

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
7. Evaluate the way a buoy would run: slide the model over whole recordings
   (the 5-minute background files and each recording's clip sequence) in
   3-second windows, and raise an alert only when several windows in a row
   score above the threshold (for example 2 of 3). Report, besides
   per-window metrics: **false alarms per hour** on background audio,
   **share of vessel passes detected** (per pass, not per clip), and
   recall separately for **near / medium / far** and for all-far-and-weak
   clips.
8. Save the run record, metrics, confusion matrix, plots and checkpoint hash.

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

**Source.** The OneDrive of Xiaoyang Du (UCAS account `duxiaoyang22`), first
author of the dataset paper (Shanghai Acoustics Laboratory, CAS): folder
`QiandaoEar22`, last modified 2024-03-20. The official repository,
https://github.com/xiaoyangdu22/QiandaoEar22, names this OneDrive as the
download, so it is the **official distribution** (checked 2026-11-04). Downloaded
2026-10-30 to 2026-11-03 as 17 zips into `data/archives/` (gitignored); names,
sizes and SHA-256 hashes are in `results/audit/archives.csv`. OneDrive could
not zip `SpeedBoat/other` in one piece, so it came in 9 parts
(`Speedboat_Other_1…9.zip`, overlapping by 177 identical files).

| OneDrive folder | Files | Distinct content not in an earlier folder |
| --- | --- | --- |
| `KaiYuan数据集/other`, `/target` | 4,000 + 1,947 | 3,988 + 1,947 |
| `SpeedBoat数据集/other`, `/target` | 6,262 + 6,927 | 2,126 + 2,526 |
| `UUV数据集/other`, `/target` | 2,000 + 902 | 0 + 0 |
| `noise_target数据集/noise/DATA0001`, `/DATA0002` | 30 + 20 | 50 |
| `noise_target数据集/target` | 4,250 | 0 |

**What the folders are** (from the paper: Du & Hong, "QiandaoEar22: A high
quality noise dataset for identifying specific ship from multiple underwater
acoustic targets using ship-radiated noise", arXiv:2406.04354, 2024). Each
folder is one of the authors' experiments. `noise_target` is Task 1 (ship vs.
background, our task): "the first 5000 ship noise recordings" plus "the
initial 30 recordings from DATA0001 and 20 recordings from DATA0002"; the
other ~18 h of background described in the paper were not published.
`SpeedBoat`, `KaiYuan` and `UUV` are Task 2 (one vessel vs. the rest); their
target counts match the paper's Table 5 exactly. The paper's SpeedBoat
training set has 2,578 "other" clips, exactly the number of "副本" copies, so
the copies are the authors' own duplication of that class. The paper counts
143 "recordings" by consistent labels, not by file timestamp, which explains
our 130. It does not define `ArtificialSignals`, `Unknown` or `KaiYan`.
Local copies of the paper and documentation are in `training/references/`
(gitignored).

**Development sample.** 27 WAV files (330 MB) from the same OneDrive in
`data/qiandaoear22/`, used by the spectrogram explorer and quick tests. All 27
are also in the full download. Inspected 2026-10-30:

| Folder | Files | Format | Notes |
| --- | --- | --- | --- |
| `speedboat_target` | 10 | 52,734 Hz, mono, 32-bit, 3.0 s | One clip each from 10 timestamps |
| `speedboat_other` | 10 | 52,734 Hz, mono, 32-bit, 3.0 s | 6 are named "… - 副本" ("copy"). Their originals are not in this sample, but the audit found them in other folders of the full download |
| `noise/data0001`, `noise/data0002` | 7 | 52,734 Hz, mono, 24-bit, 300.9 s | Full 5-minute recordings, not 3-second records |

**How this differs from the official documentation** (`QiandaoEar22_documentations.pdf`):

- The documentation says 16-bit samples; these files are 32-bit (ship clips)
  and 24-bit (noise).
- The documentation describes background noise as 25,900 three-second records
  (21 h 58 min); the OneDrive holds 50 five-minute files (4.2 h).
- The documentation's file-name example includes a `DATA0008` field that the
  ship clip names do not have.
- The documentation describes about 209 archives of roughly 100 MB; the
  OneDrive has folders of WAV files that overlap heavily (see the table).

**Open:** find the license. Neither the documentation, the paper nor the
GitHub repository states one. Until it is known, commit only metadata (names,
formats, hashes, splits), never audio, excerpts or the PDFs.

**Hydrophone facts that affect modeling** (from the documentation):
DigitalHyd SR-1, 52,734 Hz, usable band **1–25.8 kHz**. Below 1 kHz the
response is outside the specified band, so low-frequency levels are not
calibrated. Resampling to 32 kHz also discards 16–25.8 kHz.

## Known issues to fix

- ~~`clips.py` skips every "副本" file as a duplicate.~~ Fixed 2026-11-03
  (commit `8ce184b`): a copy is skipped only when byte-identical to its
  original in the same folder, and keeps its recording timestamp. In the full
  dataset every copy is a duplicate of a clip in another folder; the audit
  removes those by hash across folders.
- `backend/tests/test_language_models_paper_facts.py` (not acoustics) fails on
  Windows: it reads a JSON file without `encoding="utf-8"`. Tell the owner of
  that module.

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
