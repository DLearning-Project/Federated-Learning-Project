"""Environment configuration. Values come from environment variables or backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Federated Learning Explorer API"
    api_prefix: str = "/api/v1"
    # Comma-separated string so it can be set from a single env var.
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Folder with the locally downloaded QiandaoEar22 audio. Empty = the default
    # location inside the repository (gitignored). The app runs without it.
    acoustics_data_dir: str = ""

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def acoustics_data_path(self) -> Path:
        if self.acoustics_data_dir:
            return Path(self.acoustics_data_dir)
        repo_root = Path(__file__).resolve().parents[3]  # backend/app/core/config.py -> repo root
        return repo_root / "training" / "acoustics" / "data" / "qiandaoear22"


@lru_cache
def get_settings() -> Settings:
    return Settings()
