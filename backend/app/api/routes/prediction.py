from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.models import SUPPORTED_MODELS
from app.schemas.prediction import (
    ModelType,
    ModelsListResponse,
    ModelInfo,
    PredictionResponse,
)
from app.services.inference_service import get_inference_service
from app.services.model_registry import get_registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    settings = get_settings()
    registry = get_registry()
    models = [ModelInfo(**m) for m in registry.list_models()]
    return ModelsListResponse(models=models, default_model=settings.default_model)


prediction_router = APIRouter(prefix="/predict", tags=["prediction"])


@prediction_router.post("", response_model=PredictionResponse)
async def predict_bone_age(
    file: UploadFile = File(..., description="Hand/wrist X-ray image"),
    model_type: ModelType = Form(ModelType.cnn),
    gender: str | None = Form(None, description="Required for multimodal_cnn: male or female"),
) -> PredictionResponse:
    if model_type.value not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_type}")

    meta = get_registry().list_models()
    model_info = next(m for m in meta if m["id"] == model_type.value)
    if model_info["requires_gender"] and gender not in ("male", "female"):
        raise HTTPException(
            status_code=400,
            detail="Gender (male/female) is required for multimodal_cnn model",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        result = get_inference_service().predict(
            content, model_type.value, gender
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(**result)
