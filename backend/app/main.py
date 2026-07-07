from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analysis import router as analysis_router
from app.api.routes.health import router as health_router
from app.api.routes.prediction import prediction_router, router as models_router
from app.core.config import get_settings
from app.core.dependencies import ensure_directories


def create_app() -> FastAPI:
    settings = get_settings()
    ensure_directories()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Pediatric bone age regression from hand X-rays. "
            "Supports CNN, CNN+DNN, Multimodal CNN, and CNN+Random Forest models "
            "with Grad-CAM explainability."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api")
    app.include_router(models_router, prefix="/api")
    app.include_router(prediction_router, prefix="/api")
    app.include_router(analysis_router, prefix="/api")

    return app


app = create_app()
