from fastapi import APIRouter, HTTPException

from app.core.config import get_settings

from . import service
from .clips import ClipNotFound
from .dsp import UnsupportedAudio
from .schemas import ClipList, FedAvgInput, FedAvgResponse, SpectrogramInput, SpectrogramResponse

router = APIRouter(prefix="/acoustics", tags=["acoustics"])


@router.post("/aggregate", response_model=FedAvgResponse)
def aggregate(body: FedAvgInput) -> FedAvgResponse:
    return service.aggregate(body)


@router.get("/clips", response_model=ClipList)
def clips() -> ClipList:
    """Recordings available on this machine (empty if the dataset is not downloaded)."""
    return service.list_clips(get_settings().acoustics_data_path)


@router.post("/spectrogram", response_model=SpectrogramResponse)
def spectrogram(body: SpectrogramInput) -> SpectrogramResponse:
    try:
        return service.spectrogram(body, get_settings().acoustics_data_path)
    except ClipNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnsupportedAudio as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
