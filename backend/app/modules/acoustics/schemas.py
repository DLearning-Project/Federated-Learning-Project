from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import CalcEnvelope, StrictModel

ParamValue = Annotated[float, Field(ge=-1e6, le=1e6)]


class ClientUpdate(StrictModel):
    name: str = Field(..., min_length=1, max_length=40)
    sample_count: int = Field(..., ge=0, le=10_000_000, description="Local training samples n_k")
    vector: list[ParamValue] = Field(..., min_length=1, max_length=8, description="Toy parameter vector")


class FedAvgInput(StrictModel):
    clients: list[ClientUpdate] = Field(..., min_length=2, max_length=10)

    @model_validator(mode="after")
    def check_clients(self) -> "FedAvgInput":
        if sum(c.sample_count for c in self.clients) <= 0:
            raise ValueError("total sample count must be greater than zero")
        lengths = {len(c.vector) for c in self.clients}
        if len(lengths) != 1:
            raise ValueError("all client vectors must have the same length")
        return self


class ClientWeight(BaseModel):
    name: str
    sample_count: int
    weight: float


class FedAvgResult(BaseModel):
    total_samples: int
    weights: list[ClientWeight]
    global_vector: list[float]
    unweighted_mean: list[float]


FedAvgResponse = CalcEnvelope[FedAvgInput, FedAvgResult]


# --- Spectrogram explorer: real audio -> log-Mel, computed on request -------------


class SpectrogramInput(StrictModel):
    """Audio source plus the front-end settings the sliders control."""

    source: Literal["synthetic", "recording"] = "synthetic"
    clip_id: str | None = Field(None, max_length=400, description="Clip id from GET /acoustics/clips")
    sample_rate_hz: Literal[16000, 32000] = Field(32000, description="Rate the audio is resampled to")
    n_fft: Literal[256, 512, 1024, 2048, 4096] = 1024
    hop_length: int = Field(320, ge=128, le=2048, description="Samples between STFT frames")
    n_mels: int = Field(64, ge=16, le=128)
    fmin_hz: float = Field(50.0, ge=0.0, le=8000.0)
    fmax_hz: float = Field(14000.0, ge=500.0, le=16000.0)

    @model_validator(mode="after")
    def check_settings(self) -> "SpectrogramInput":
        if self.source == "recording" and not self.clip_id:
            raise ValueError("clip_id is required when source is 'recording'")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must not exceed n_fft")
        if self.fmin_hz >= self.fmax_hz:
            raise ValueError("fmin_hz must be below fmax_hz")
        if self.fmax_hz > self.sample_rate_hz / 2:
            raise ValueError("fmax_hz must not exceed half the sample rate")
        return self


class StageTimings(BaseModel):
    read_ms: float
    resample_ms: float
    stft_ms: float
    mel_ms: float
    total_ms: float


class SpectrogramResult(BaseModel):
    source_label: str
    recording_id: str | None
    original_sample_rate_hz: int
    clip_seconds: float
    rms_dbfs: float
    frames: int
    mel_bins: int
    frame_seconds: float
    empty_mel_filters: int
    db_max: float
    db_min_shown: float
    mel_center_hz: list[float]
    spectrogram_db: list[list[float]]  # [mel bin, low to high frequency][frame]
    timings: StageTimings


SpectrogramResponse = CalcEnvelope[SpectrogramInput, SpectrogramResult]


class ClipInfo(BaseModel):
    id: str
    label: str
    category: Literal["speedboat_target", "speedboat_other", "noise"]
    recording_id: str


class ClipList(BaseModel):
    dataset_found: bool
    folder_hint: str
    clips: list[ClipInfo]
