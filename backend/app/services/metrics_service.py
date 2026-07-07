"""Metrics loading and comparison service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.models import MODEL_REGISTRY


def _metrics_path() -> Path:
    return get_settings().metrics_dir / "evaluation_results.json"


def load_evaluation_results() -> dict[str, Any]:
    path = _metrics_path()
    if not path.exists():
        return _default_metrics()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default_metrics() -> dict[str, Any]:
    """Sample metrics for demo when training has not been run."""
    models = []
    sample = [
        ("cnn", 8.42, 98.76, 9.94),
        ("cnn_dnn", 7.89, 86.54, 9.30),
        ("multimodal_cnn", 7.15, 72.18, 8.49),
        ("cnn_rf", 8.01, 89.32, 9.45),
    ]
    for model_id, mae, mse, rmse in sample:
        meta = MODEL_REGISTRY[model_id]
        models.append(
            {
                "model_type": model_id,
                "display_name": meta["display_name"],
                "metrics": {"mae": mae, "mse": mse, "rmse": rmse},
                "num_samples": 1425,
            }
        )
    best = min(models, key=lambda m: m["metrics"]["mae"])
    return {
        "comparison": {
            "models": models,
            "best_model": best["model_type"],
            "generated_at": "sample-data",
        },
        "plots": {},
        "details": {"note": "Sample metrics — run ml.evaluate after training for real values"},
    }


def get_comparison_table() -> dict:
    data = load_evaluation_results()
    return data["comparison"]


def get_plot_path(plot_name: str) -> Path | None:
    settings = get_settings()
    candidates = [
        settings.metrics_dir / "plots" / f"{plot_name}.png",
        settings.metrics_dir / f"{plot_name}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None
