# Federated Learning Explorer

An educational web app that explains federated learning through three independent modules: **language models** (LoRA / FLoRIST-inspired fine-tuning), **underwater acoustics** (spectrograms, EfficientNet-B0 concepts, FedAvg) and **hardware-aware learning** (device time/energy tradeoffs). It has a React + TypeScript frontend and a FastAPI backend that runs small, bounded calculations.

**Current MVP limitations:** every number is *illustrative*. There is no real model training, no dataset download, no GPU use, no database and no authentication. Device profiles are uncalibrated. Settings and results live only in browser memory (the server stores nothing): they survive moving between pages but reset when you refresh.

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
    acoustics/        page, spectrogram + stacked bars, static sample JSON
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
| Underwater acoustics | `frontend/src/modules/acoustics` | `backend/app/modules/acoustics` | `POST /api/v1/acoustics/aggregate` | Bryn |
| Hardware-aware learning | `frontend/src/modules/hardware` | `backend/app/modules/hardware` | `GET /api/v1/hardware/device-profiles`, `POST /api/v1/hardware/estimate` | Sai |

Shared files (`app/`, `components/`, `services/`, `styles/`, `backend/app/main.py`, `schemas/common.py`) change only after agreeing as a team.

## Add to your module

1. **Schema**: add or extend Pydantic models in `backend/app/modules/<name>/schemas.py`. Bound every field and use `StrictModel`.
2. **Service**: put the calculation in `service.py`, return a `CalcEnvelope` with `units` and `assumptions`. Heavy work (real training) belongs in a separate worker later, not in the request.
3. **Router**: expose it in `router.py`; it should only call the service.
4. **Frontend**: mirror the types in `modules/<name>/types.ts` (same field names), add the call in `modules/<name>/api.ts`, and use `useApiAction` for loading/error states and ignoring stale responses. Don't re-implement backend formulas in the UI.
5. **Verify**: add a hand-calculated test in `backend/tests/`, then run `pytest` (in `backend/`, after `pip install -r requirements-dev.txt`) and `npm run build` (in `frontend/`).

Labelled "Extension point" cards on each page mark where real models (FLoRIST/SVT, PORTIA + EfficientNet-B0, device calibration) will plug in.

## Environment settings

| File | Variable | Default | Purpose |
| --- | --- | --- | --- |
| `frontend/.env` | `VITE_API_BASE_URL` | `http://localhost:8000` | Where the frontend sends API calls (public; never put secrets here) |
| `backend/.env` | `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated CORS origins |

**Backend connection troubleshooting:** if the sidebar shows "Backend offline" or a module says "Backend not reachable", check that `uvicorn` is running on port 8000 and that http://localhost:8000/api/v1/health returns `{"status":"ok","version":"0.1.0"}`. If the frontend runs on a different host or port, add it to `ALLOWED_ORIGINS` and restart the backend. If the API runs elsewhere, set `VITE_API_BASE_URL` and restart `npm run dev`. Lessons and quizzes keep working while the backend is down; once it is back, press **Retry** in the module (or **Check again** in the sidebar).
