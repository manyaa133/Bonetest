from pathlib import Path

from app.core.config import get_settings


def ensure_directories() -> None:
    settings = get_settings()
    for directory in (
        settings.checkpoint_dir,
        settings.metrics_dir,
        settings.upload_dir,
        settings.rf_models_dir,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
