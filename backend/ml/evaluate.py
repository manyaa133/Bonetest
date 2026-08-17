"""Evaluate all trained models on validation set.

Usage:
    python -m ml.evaluate --data-dir /path/to/rsna --checkpoints-dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from app.models.new_models import MultiscaleCNNFeatureExtractor, MultiscaleCNNRFWrapper
from ml.dataset import create_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bone age models")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=BACKEND_ROOT / "checkpoints"
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=BACKEND_ROOT / "metrics"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}


@torch.no_grad()
def evaluate_model(model_type: str, loader, device, checkpoints_dir: Path):
    meta = MODEL_REGISTRY[model_type]
    ckpt_path = checkpoints_dir / f"{model_type}_best.pt"
    if not ckpt_path.exists():
        print(f"Skipping {model_type}: checkpoint not found")
        return None, None, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if model_type in {"cnn_rf", "multiscale_cnn_rf"}:
        if model_type == "cnn_rf":
            cnn = CNNFeatureExtractor(pretrained=False)
            wrapper_cls = CNNWithRFWrapper
            rf_path = BACKEND_ROOT / "rf_models" / "cnn_rf.joblib"
        else:
            cnn = MultiscaleCNNFeatureExtractor(pretrained=False)
            wrapper_cls = MultiscaleCNNRFWrapper
            rf_path = BACKEND_ROOT / "rf_models" / "multiscale_cnn_rf.joblib"
        cnn.load_state_dict(checkpoint["model_state_dict"], strict=False)
        cnn.to(device).eval()
        rf = joblib.load(rf_path)
        wrapper = wrapper_cls(cnn, rf)
        preds, targets, ids = [], [], []
        for batch in loader:
            imgs = batch["image"].to(device)
            feats = cnn(imgs).cpu().numpy()
            batch_preds = rf.predict(feats)
            preds.extend(batch_preds.tolist())
            targets.extend(batch["bone_age"].numpy().tolist())
            ids.extend(batch["id"])
        return np.array(targets), np.array(preds), ids

    model = meta["class"](pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()

    preds, targets, ids = [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        if model_type in {"multimodal_cnn", "regression_multimodal"}:
            gender = batch["male"].unsqueeze(-1).to(device)
            out = model(imgs, gender)
        else:
            out = model(imgs)
        preds.extend(out.cpu().numpy().tolist())
        targets.extend(batch["bone_age"].numpy().tolist())
        ids.extend(batch["id"])

    return np.array(targets), np.array(preds), ids


def plot_comparison(results: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}
    sns.set_theme(style="whitegrid")

    models = list(results.keys())
    metrics = ["mae", "mse", "rmse"]
    labels = ["MAE (months)", "MSE", "RMSE (months)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, label in zip(axes, metrics, labels):
        values = [results[m]["metrics"][metric] for m in models]
        names = [results[m]["display_name"] for m in models]
        bars = ax.bar(names, values, color=sns.color_palette("muted", len(models)))
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    comparison_path = output_dir / "metrics_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["metrics_comparison"] = str(comparison_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    for model_type, data in results.items():
        df = pd.read_csv(data["predictions_csv"])
        ax.scatter(
            df["actual"],
            df["predicted"],
            alpha=0.3,
            s=12,
            label=data["display_name"],
        )
    lims = [0, 220]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Bone Age (months)")
    ax.set_ylabel("Predicted Bone Age (months)")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=8)
    scatter_path = output_dir / "scatter_comparison.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["scatter_comparison"] = str(scatter_path)

    return plot_paths


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, _, val_df = create_dataloaders(args.data_dir, batch_size=32)

    all_results = {}
    for model_type in SUPPORTED_MODELS:
        print(f"Evaluating {model_type}...")
        y_true, y_pred, ids = evaluate_model(
            model_type, val_loader, device, args.checkpoints_dir
        )
        if y_true is None:
            continue

        metrics = compute_metrics(y_true, y_pred)
        meta = MODEL_REGISTRY[model_type]
        pred_csv = args.metrics_dir / f"predictions_{model_type}.csv"
        pd.DataFrame(
            {"id": ids, "actual": y_true, "predicted": y_pred}
        ).to_csv(pred_csv, index=False)

        all_results[model_type] = {
            "model_type": model_type,
            "display_name": meta["display_name"],
            "metrics": metrics,
            "num_samples": len(y_true),
            "predictions_csv": str(pred_csv),
        }
        print(f"  MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

    if not all_results:
        print("No checkpoints found to evaluate.")
        return

    best = min(all_results.items(), key=lambda x: x[1]["metrics"]["mae"])
    plots_dir = args.metrics_dir / "plots"
    plot_paths = plot_comparison(all_results, plots_dir)

    output = {
        "comparison": {
            "models": [
                {
                    "model_type": k,
                    "display_name": v["display_name"],
                    "metrics": v["metrics"],
                    "num_samples": v["num_samples"],
                }
                for k, v in all_results.items()
            ],
            "best_model": best[0],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }

    out_path = args.metrics_dir / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
