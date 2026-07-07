"""Training script for all bone age models.

Usage (Kaggle/Colab/local GPU):
    cd backend
    python -m ml.train --model-type cnn --data-dir /path/to/rsna --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY
from app.models.cnn_rf import CNNFeatureExtractor
from ml.augmentation import TrainAugmentation
from ml.dataset import create_dataloaders
from ml.preprocessing import compute_dataset_statistics, save_normalization_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bone age regression models")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "cnn_dnn", "multimodal_cnn", "cnn_rf"],
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=BACKEND_ROOT / "checkpoints"
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device, model_type, scaler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["bone_age"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            if model_type == "multimodal_cnn":
                gender = batch["male"].unsqueeze(-1).to(device)
                outputs = model(images, gender)
            else:
                outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, model_type):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    for batch in loader:
        images = batch["image"].to(device)
        y = batch["bone_age"].to(device)

        if model_type == "multimodal_cnn":
            gender = batch["male"].unsqueeze(-1).to(device)
            outputs = model(images, gender)
        else:
            outputs = model(images)

        loss = criterion(outputs, y)
        total_loss += loss.item() * images.size(0)
        preds.extend(outputs.cpu().numpy().tolist())
        targets.extend(y.cpu().numpy().tolist())

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    rmse = float(np.sqrt(mse))
    return total_loss / len(loader.dataset), mae, mse, rmse


def train_random_forest(
    data_dir: Path,
    checkpoints_dir: Path,
    cnn_checkpoint: Path,
    device: torch.device,
    image_size: int,
    norm_mean: float,
    norm_std: float,
) -> None:
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    from ml.dataset import BoneAgeDataset, stratified_split
    import pandas as pd

    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        csv_path = data_dir / "boneage-training-dataset.csv"
    df = pd.read_csv(csv_path)
    image_dir = data_dir / "train" if (data_dir / "train").exists() else data_dir
    train_df, val_df = stratified_split(df)

    cnn = CNNFeatureExtractor(pretrained=False)
    ckpt = torch.load(cnn_checkpoint, map_location=device, weights_only=False)
    cnn.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    cnn.to(device).eval()

    def extract_features(subset_df):
        ds = BoneAgeDataset(subset_df, image_dir, image_size, None, norm_mean, norm_std)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
        features, ages = [], []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                feats = cnn(imgs).cpu().numpy()
                features.append(feats)
                ages.extend(batch["bone_age"].numpy().tolist())
        return np.vstack(features), np.array(ages)

    X_train, y_train = extract_features(train_df)
    X_val, y_val = extract_features(val_df)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_val)
    mae = float(np.mean(np.abs(preds - y_val)))
    mse = float(np.mean((preds - y_val) ** 2))
    rmse = float(np.sqrt(mse))
    print(f"RF validation — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    rf_dir = BACKEND_ROOT / "rf_models"
    rf_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, rf_dir / "cnn_rf.joblib")

    torch.save(
        {
            "model_state_dict": cnn.state_dict(),
            "model_type": "cnn_rf",
            "val_mae": mae,
            "val_rmse": rmse,
        },
        checkpoints_dir / "cnn_rf_best.pt",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_dataset_statistics(args.data_dir)
    save_normalization_stats(stats, args.checkpoints_dir / "normalization.json")
    norm_mean, norm_std = stats["mean"], stats["std"]
    print(f"Normalization — mean: {norm_mean:.4f}, std: {norm_std:.4f}")

    augment = TrainAugmentation()
    train_loader, val_loader, _, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_transform=augment,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    meta = MODEL_REGISTRY[args.model_type]
    model = meta["class"](pretrained=args.pretrained).to(device)

    if args.model_type == "cnn_rf":
        # Stage 1: train CNN baseline first, then RF in separate step
        print("Training CNN feature extractor for RF pipeline...")
        # Reuse cnn architecture temporarily
        from app.models.cnn import CNNBaseline

        model = CNNBaseline(pretrained=args.pretrained).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    best_mae = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args.model_type, scaler
        )
        val_loss, mae, mse, rmse = validate(
            model, val_loader, criterion, device, args.model_type
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"train_loss: {train_loss:.4f}, val_mae: {mae:.2f}, val_rmse: {rmse:.2f}"
        )

        if mae < best_mae:
            best_mae = mae
            patience_counter = 0
            ckpt_name = f"{args.model_type}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model_type,
                    "val_mae": mae,
                    "val_mse": mse,
                    "val_rmse": rmse,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                },
                args.checkpoints_dir / ckpt_name,
            )
            print(f"  Saved checkpoint: {ckpt_name}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break

    history_path = args.checkpoints_dir / f"{args.model_type}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if args.model_type == "cnn_rf":
        cnn_ckpt = args.checkpoints_dir / "cnn_rf_best.pt"
        train_random_forest(
            args.data_dir,
            args.checkpoints_dir,
            cnn_ckpt,
            device,
            args.image_size,
            norm_mean,
            norm_std,
        )


if __name__ == "__main__":
    main()
