from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.prediction import HealthResponse
from app.services.model_registry import get_registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    registry = get_registry()
    loaded = list(registry._models.keys())
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        models_loaded=loaded,
    )
