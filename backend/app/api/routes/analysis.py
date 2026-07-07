from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.models import SUPPORTED_MODELS
from app.schemas.prediction import GradCamResponse, ModelType
from app.services.gradcam_service import get_gradcam_service
from app.services.metrics_service import get_plot_path, load_evaluation_results
from app.schemas.prediction import ComparisonResponse, EvaluationResponse

router = APIRouter(tags=["gradcam", "metrics"])


@router.post("/gradcam", response_model=GradCamResponse)
async def generate_gradcam(
    file: UploadFile = File(...),
    model_type: ModelType = Form(ModelType.cnn),
    gender: str | None = Form(None),
) -> GradCamResponse:
    if model_type.value not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_type}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        result = get_gradcam_service().generate(
            content, model_type.value, gender
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {exc}") from exc

    return GradCamResponse(**result)


@router.get("/metrics", response_model=EvaluationResponse)
def get_metrics() -> EvaluationResponse:
    data = load_evaluation_results()
    return EvaluationResponse(**data)


@router.get("/metrics/comparison", response_model=ComparisonResponse)
def get_comparison() -> ComparisonResponse:
    data = load_evaluation_results()
    return ComparisonResponse(**data["comparison"])


@router.get("/metrics/plots/{plot_name}")
def get_plot(plot_name: str) -> Response:
    path = get_plot_path(plot_name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found")
    return Response(content=path.read_bytes(), media_type="image/png")
