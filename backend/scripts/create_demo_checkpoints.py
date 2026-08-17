"""Create untrained demo checkpoints for local inference testing.

These weights are randomly initialized — predictions will not be clinically
accurate. Replace with trained checkpoints from ml.train for production use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn_rf import CNNFeatureExtractor
from app.models.new_models import MultiscaleCNNFeatureExtractor


def main() -> None:
    ckpt_dir = BACKEND_ROOT / "checkpoints"
    rf_dir = BACKEND_ROOT / "rf_models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rf_dir.mkdir(parents=True, exist_ok=True)

    for model_type in SUPPORTED_MODELS:
        if model_type in {"cnn_rf", "multiscale_cnn_rf"}:
            continue
        meta = MODEL_REGISTRY[model_type]
        model = meta["class"](pretrained=False)
        path = ckpt_dir / f"{model_type}_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "note": "Demo checkpoint — train with ml.train for real weights",
            },
            path,
        )
        print(f"Created {path}")

    # CNN + RF
    cnn = CNNFeatureExtractor(pretrained=False)
    torch.save(
        {"model_state_dict": cnn.state_dict(), "model_type": "cnn_rf"},
        ckpt_dir / "cnn_rf_best.pt",
    )

    rng = np.random.RandomState(42)
    X_dummy = rng.randn(100, 512)
    y_dummy = rng.uniform(24, 180, 100)
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_dummy, y_dummy)
    joblib.dump(rf, rf_dir / "cnn_rf.joblib")
    print(f"Created demo RF model at {rf_dir / 'cnn_rf.joblib'}")

    multiscale_cnn = MultiscaleCNNFeatureExtractor(pretrained=False)
    torch.save(
        {"model_state_dict": multiscale_cnn.state_dict(), "model_type": "multiscale_cnn_rf"},
        ckpt_dir / "multiscale_cnn_rf_best.pt",
    )
    joblib.dump(rf, rf_dir / "multiscale_cnn_rf.joblib")
    print(f"Created demo RF model at {rf_dir / 'multiscale_cnn_rf.joblib'}")
    print("\nDemo checkpoints ready. Start the API with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
