# Federated Learning Explorer

An educational web app that explains federated learning through three independent modules: **language models** (LoRA / FLoRIST-inspired fine-tuning), **underwater acoustics** (live Mel spectrograms of real ship recordings, FedAvg, and a planned PANN MobileNetV2 vessel classifier) and **hardware-aware learning** (device time/energy tradeoffs). It has a React + TypeScript frontend and a FastAPI backend that runs small, bounded calculations.

**Current status (mid-project):**

- **Real computation:** the acoustics spectrogram explorer reads real QiandaoEar22 recordings, resamples them, and computes the Mel spectrogram on the server for the exact settings you choose (labeled "measured" in the UI). The FedAvg weighting and LoRA/hardware calculators are exact formulas, not trained models.
- **Illustrative or not built yet:** no neural network has been trained or run yet (the PANN MobileNetV2 vessel classifier is planned), and the language-model fine-tuning and device profiles are estimates. Anything labeled "illustrative" in the UI is not a measurement.
- **Other limits:** no database and no authentication. Settings and results live only in browser memory (the server stores nothing): they survive moving between pages but reset when you refresh.

## Setup

Prerequisites: **Node.js 18+** (tested with 20) and **Python 3.10+**.

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # optional; defaults work for local dev
uvicorn app.main:app --reload --port 8000
```

### Frontend (in a second terminal)

```bash
cd frontend
npm install
cp .env.example .env                 # optional; defaults to http://localhost:8000
npm run dev
```

### Try it (about 1 minute)

1. Start the backend and frontend as above and open http://localhost:5173.
2. Open the **Underwater Acoustics** module and find **What the model hears**.
3. Change the **Preset** (for example PANN default, then Engine tones). The backend recomputes the spectrogram, and the model input size and server time update.
4. The dataset is not stored in this repository (it is about 21 GB). Without it, the **Audio** menu offers a clearly labeled synthetic signal instead of real recordings. To use real audio, download QiandaoEar22 from [IEEE DataPort](https://ieee-dataport.org/documents/qiandaoear22), place the WAV files under `training/acoustics/data/qiandaoear22`, or point `ACOUSTICS_DATA_DIR` in `backend/.env` at them.
5. Run the backend tests with `pytest` in `backend/` (after `pip install -r requirements-dev.txt`).

### Local URLs

| What | URL |
| --- | --- |
| Frontend | http://localhost:5173 |
| API | http://localhost:8000/api/v1 |
| API docs (Swagger) | http://localhost:8000/docs |

## Project layout

```text
frontend/src/
  app/              layout, routing, module registry (modules.ts), shared state, homepage
  components/       shared UI: Card, NumberField, Quiz, StatusMessage, Results, FederatedDiagram
  hooks/            useApiAction (request lifecycle, cancels stale requests)
  services/api.ts   API client and error handling
  modules/
    language-models/  page, matrix visual, types, API calls
    acoustics/        page, spectrogram explorer, FedAvg calculator
    hardware/         page, resource bars, types, API calls
  styles/           tokens.css (colors from colors.md, spacing, type), global.css
backend/app/
  main.py           FastAPI app, CORS, /health, /modules, router registration
  core/config.py    environment settings
  schemas/common.py shared CalcEnvelope response (mode, inputs, results, units, assumptions)
  modules/<name>/   router.py (HTTP), schemas.py (validation), service.py (calculations)
  data/             illustrative device_profiles.json
backend/tests/      pytest checks for calculations and invalid inputs
```

### Module ownership

| Module | Frontend folder | Backend package | Endpoint(s) | Owner |
| --- | --- | --- | --- | --- |
| Language models | `frontend/src/modules/language-models` | `backend/app/modules/language_models` | `POST /api/v1/language-models/calculate` | Nhi |
| Underwater acoustics | `frontend/src/modules/acoustics` | `backend/app/modules/acoustics` | `POST /api/v1/acoustics/aggregate`, `GET /api/v1/acoustics/clips`, `POST /api/v1/acoustics/spectrogram` | Bryn |
| Hardware-aware learning | `frontend/src/modules/hardware` | `backend/app/modules/hardware` | `GET /api/v1/hardware/device-profiles`, `POST /api/v1/hardware/estimate` | Sai |

Shared files (`app/`, `components/`, `services/`, `styles/`, `backend/app/main.py`, `schemas/common.py`) change only after agreeing as a team.

## Add to your module

1. **Schema**: add or extend Pydantic models in `backend/app/modules/<name>/schemas.py`. Bound every field and use `StrictModel`.
2. **Service**: put the calculation in `service.py`, return a `CalcEnvelope` with `units` and `assumptions`. Heavy work (real training) belongs in a separate worker later, not in the request.
3. **Router**: expose it in `router.py`; it should only call the service.
4. **Frontend**: mirror the types in `modules/<name>/types.ts` (same field names), add the call in `modules/<name>/api.ts`, and use `useApiAction` for loading/error states and ignoring stale responses. Don't re-implement backend formulas in the UI.
5. **Verify**: add a hand-calculated test in `backend/tests/`, then run `pytest` (in `backend/`, after `pip install -r requirements-dev.txt`) and `npm run build` (in `frontend/`).

Labelled "Extension point" cards on each page mark where real models (FLoRIST/SVT, PANN MobileNetV2 trained on QiandaoEar22, device calibration) will plug in.

## Offline model training (acoustics)

Training runs outside the web app, never inside an API request. Its setup, plan and progress are in [`training/acoustics/README.md`](training/acoustics/README.md) and [`training/acoustics/TRAINING.md`](training/acoustics/TRAINING.md). It needs Python 3.12 and an NVIDIA GPU, and is not required to run the web app.

## Environment settings

| File | Variable | Default | Purpose |
| --- | --- | --- | --- |
| `frontend/.env` | `VITE_API_BASE_URL` | `http://localhost:8000` | Where the frontend sends API calls (public; never put secrets here) |
| `backend/.env` | `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated CORS origins |

**Backend connection troubleshooting:** if the sidebar shows "Backend offline" or a module says "Backend not reachable", check that `uvicorn` is running on port 8000 and that http://localhost:8000/api/v1/health returns `{"status":"ok","version":"0.1.0"}`. If the frontend runs on a different host or port, add it to `ALLOWED_ORIGINS` and restart the backend. If the API runs elsewhere, set `VITE_API_BASE_URL` and restart `npm run dev`. Lessons and quizzes keep working while the backend is down; once it is back, press **Retry** in the module (or **Check again** in the sidebar).
