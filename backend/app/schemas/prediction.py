from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    cnn = "cnn"
    cnn_dnn = "cnn_dnn"
    multimodal_cnn = "multimodal_cnn"
    cnn_rf = "cnn_rf"


class Gender(str, Enum):
    male = "male"
    female = "female"


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]


class ModelInfo(BaseModel):
    id: str
    display_name: str
    description: str
    requires_gender: bool
    supports_gradcam: bool
    checkpoint_available: bool


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str


class PredictionResponse(BaseModel):
    model_type: str
    bone_age_months: float
    confidence: float = Field(ge=0.0, le=1.0)
    gender_used: Optional[str] = None
    processing_time_ms: float


class GradCamResponse(BaseModel):
    model_type: str
    bone_age_months: float
    confidence: float
    heatmap_base64: str
    overlay_base64: str
    processing_time_ms: float


class MetricValues(BaseModel):
    mae: float
    mse: float
    rmse: float


class ModelMetrics(BaseModel):
    model_type: str
    display_name: str
    metrics: MetricValues
    num_samples: int


class ComparisonResponse(BaseModel):
    models: List[ModelMetrics]
    best_model: str
    generated_at: str


class EvaluationResponse(BaseModel):
    comparison: ComparisonResponse
    plots: Dict[str, str]
    details: Optional[Dict] = None


class ErrorResponse(BaseModel):
    detail: str
