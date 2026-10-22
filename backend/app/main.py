"""FastAPI entry point: CORS, error formatting, shared routes and module routers."""

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.modules.acoustics.router import router as acoustics_router
from app.modules.hardware.router import router as hardware_router
from app.modules.language_models.router import router as language_models_router
from app.schemas.common import HealthResponse, ModuleInfo

VERSION = "0.1.0"
settings = get_settings()

app = FastAPI(title=settings.app_name, version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Flatten Pydantic errors into readable messages for the UI."""
    errors = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p != "body"]
        msg = str(err.get("msg", "Invalid value")).removeprefix("Value error, ")
        errors.append({"field": ".".join(loc) or None, "message": msg})
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(e["message"] for e in errors), "errors": errors},
    )


MODULES = [
    ModuleInfo(
        id="language-models",
        title="Language models",
        description="LoRA adapters and FLoRIST-inspired federated fine-tuning: what is trained and what is sent.",
        path="/language-models",
    ),
    ModuleInfo(
        id="acoustics",
        title="Underwater acoustics",
        description="Spectrograms, EfficientNet-B0 concepts, and sample-weighted FedAvg aggregation.",
        path="/acoustics",
    ),
    ModuleInfo(
        id="hardware",
        title="Hardware-aware learning",
        description="Device time, energy and bandwidth tradeoffs for federated clients.",
        path="/hardware",
    ),
]

api = APIRouter(prefix=settings.api_prefix)


@api.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=VERSION)


@api.get("/modules", response_model=list[ModuleInfo])
def modules() -> list[ModuleInfo]:
    return MODULES


api.include_router(language_models_router)
api.include_router(acoustics_router)
api.include_router(hardware_router)
app.include_router(api)
