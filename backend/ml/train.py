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

import time 
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY
from app.models.cnn_rf import CNNFeatureExtractor
from app.models.new_models import MultiscaleCNNFeatureExtractor
from ml.augmentation import TrainAugmentation
from ml.dataset import create_dataloaders
from ml.preprocessing import compute_dataset_statistics, save_normalization_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bone age regression models")
    parser.add_argument(
        "--model-type",
        choices=[
            "cnn",
            "cnn_dnn",
            "multimodal_cnn",
            "regression_multimodal",
            "mask_rcnn",
            "ensemble_cnn",
            "cnn_tw3",
            "cnn_rf",
            "multiscale_cnn_rf",
        ],
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
    parser.add_argument(
        "--cnn-checkpoint",
        type=Path,
        default=None,
        help="Path to a pretrained CNN checkpoint",
    )
    return parser.parse_args()


def _model_requires_gender(model_type: str) -> bool:
    return model_type in {"multimodal_cnn", "regression_multimodal"}


def train_epoch(model, loader, optimizer, criterion, device, model_type, scaler):
    model.train()
    total_loss = 0.0

    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):

        if batch_idx % 50 == 0:
            print(f"Training batch {batch_idx+1}/{num_batches}")

        images = batch["image"].to(device)
        targets = batch["bone_age"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):

            if _model_requires_gender(model_type):
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

        if _model_requires_gender(model_type):
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
    device: torch.device,
    image_size: int,
    norm_mean: float,
    norm_std: float,
    model_type: str,
) -> None:

    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    from ml.dataset import (
        BoneAgeDataset,
        stratified_split,
        find_image_directory,
    )

    # ----------------------------
    # Locate CSV
    # ----------------------------
    csv_path = data_dir / "train.csv"

    if not csv_path.exists():
        csv_path = data_dir / "boneage-training-dataset.csv"

    if not csv_path.exists():
        raise FileNotFoundError("Training CSV not found.")

    df = pd.read_csv(csv_path)

    # ----------------------------
    # Locate image folder
    # ----------------------------
    image_dir = find_image_directory(data_dir)

    print(f"\nUsing image directory: {image_dir}\n")

    train_df, val_df = stratified_split(df)

    # ----------------------------
    # Load pretrained CNN or initialize deterministic feature extractor
    # ----------------------------
    if model_type == "cnn_rf":
        cnn_checkpoint = checkpoints_dir / "cnn_best.pt"

        if not cnn_checkpoint.exists():
            raise FileNotFoundError(
                f"Cannot find pretrained CNN:\n{cnn_checkpoint}"
            )

        print(f"Loading pretrained CNN from {cnn_checkpoint}")
        cnn = CNNFeatureExtractor(pretrained=False)

        checkpoint = torch.load(
            cnn_checkpoint,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )

        cnn.load_state_dict(
            state_dict,
            strict=False,
        )
    else:
        print("Initializing deterministic multiscale feature extractor")
        torch.manual_seed(42)
        cnn = MultiscaleCNNFeatureExtractor(pretrained=False)

    cnn.to(device)
    cnn.eval()

    # ----------------------------
    # Feature extraction
    # ----------------------------
    def extract_features(subset_df):

        dataset = BoneAgeDataset(
            subset_df,
            image_dir,
            image_size,
            None,
            norm_mean,
            norm_std,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        features = []
        ages = []

        with torch.no_grad():

            for i, batch in enumerate(loader):

                if i % 50 == 0:
                    print(
                        f"Extracting features: batch {i+1}/{len(loader)}"
                    )

                images = batch["image"].to(device)

                feats = cnn(images)

                feats = feats.cpu().numpy()

                features.append(feats)

                ages.extend(
                    batch["bone_age"].numpy().tolist()
                )

        return np.vstack(features), np.array(ages)

    print("\nExtracting training features...")
    X_train, y_train = extract_features(train_df)

    print("\nExtracting validation features...")
    X_val, y_val = extract_features(val_df)

    # ----------------------------
    # Train Random Forest
    # ----------------------------
    print("\nTraining Random Forest...")
    start = time.time()

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    print(f"Random Forest training finished in {(time.time()-start)/60:.2f} minutes")

    # ----------------------------
    # Evaluate
    # ----------------------------
    preds = rf.predict(X_val)

    mae = float(np.mean(np.abs(preds - y_val)))
    mse = float(np.mean((preds - y_val) ** 2))
    rmse = float(np.sqrt(mse))

    print(f"\nRF Validation")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # ----------------------------
    # Save RF model
    # ----------------------------
    rf_dir = BACKEND_ROOT / "rf_models"
    rf_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        rf,
        rf_dir / f"{model_type}.joblib",
    )

    # ----------------------------
    # Save combined checkpoint
    # ----------------------------
    torch.save(
        {
            "model_state_dict": cnn.state_dict(),
            "model_type": model_type,
            "val_mae": mae,
            "val_rmse": rmse,
        },
        checkpoints_dir / f"{model_type}_best.pt",
    )

    print("\nRandom Forest training completed.")
def main() -> None:

    # -------------------------
    # Step 1
    # -------------------------
    print("Step 1: Parsing arguments")
    args = parse_args()

    # -------------------------
    # Step 2
    # -------------------------
    print("Step 2: Setting device")
    device = torch.device(args.device)

    # -------------------------
    # Step 3
    # -------------------------
    print("Step 3: Creating checkpoint directory")
    args.checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Step 4
    # -------------------------
    print("Step 4: Using fixed normalization values")

    norm_mean = 0.4523
    norm_std = 0.2118

    stats = {
        "mean": norm_mean,
        "std": norm_std,
    }

    save_normalization_stats(
        stats,
        args.checkpoints_dir / "normalization.json",
    )

    print(
        f"Normalization — mean: {norm_mean:.4f}, std: {norm_std:.4f}"
    )

    # ======================================================
    # CNN + Random Forest
    # Skip CNN training completely
    # ======================================================

    if args.model_type in {"cnn_rf", "multiscale_cnn_rf"}:

        print("\nUsing existing CNN checkpoint (cnn_best.pt)")
        print("Skipping CNN training...")
        print("Starting Random Forest training...\n")

        train_random_forest(
            args.data_dir,
            args.checkpoints_dir,
            device,
            args.image_size,
            norm_mean,
            norm_std,
            args.model_type,
        )

        print("Training completed successfully!")
        return

    # -------------------------
    # Step 5
    # -------------------------
    print("Step 5: Creating augmentations")

    augment = TrainAugmentation()

    # -------------------------
    # Step 6
    # -------------------------
    print("Step 6: Creating dataloaders")

    train_loader, val_loader, _, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_transform=augment,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    # -------------------------
    # Step 7
    # -------------------------
    print("Step 7: Creating model")

    meta = MODEL_REGISTRY[args.model_type]

    model = meta["class"](
        pretrained=args.pretrained
    ).to(device)
    # -------------------------------------------------
    # Load pretrained CNN backbone into multimodal model
    # -------------------------------------------------
    # -------------------------------------------------
    # Load pretrained CNN backbone
    # -------------------------------------------------
    if args.model_type in [
        "multimodal_cnn",
        "regression_multimodal",
        "cnn_dnn",
        "cnn_tw3",
    ] and args.cnn_checkpoint is not None:
        print(f"\nLoading pretrained CNN from {args.cnn_checkpoint}")

        checkpoint = torch.load(
            args.cnn_checkpoint,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )

        backbone_state = {}

        # Extract only CNN backbone weights
        for key, value in state_dict.items():

            if key.startswith("backbone."):
                new_key = key.replace(
                    "backbone.",
                    "image_backbone.",
                    1
                )

            elif key.startswith("image_backbone."):
                new_key = key

            else:
                continue

            backbone_state[new_key] = value

        # Load pretrained CNN backbone
        missing, unexpected = model.load_state_dict(
            backbone_state,
            strict=False,
        )

        print("CNN backbone loaded successfully!")

        # Freeze pretrained CNN backbone
        if args.model_type in {
            "multimodal_cnn",
            "cnn_tw3",
        }:
            for p in model.image_backbone.parameters():
                p.requires_grad = False

        elif args.model_type == "cnn_dnn":
            for p in model.backbone.parameters():
                p.requires_grad = False

        print("CNN backbone frozen. Only the additional branches will be trained.")
    # -------------------------
    # Step 8
    # -------------------------
    print("Step 8: Creating optimizer")

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = GradScaler(
        enabled=device.type == "cuda"
    )

    best_mae = float("inf")
    patience_counter = 0
    history = []

    # -------------------------
    # Step 9
    # -------------------------
    print("Step 9: Starting training")

    for epoch in range(1, args.epochs + 1):

        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.model_type,
            scaler,
        )

        val_loss, mae, mse, rmse = validate(
            model,
            val_loader,
            criterion,
            device,
            args.model_type,
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
            f"train_loss={train_loss:.4f} | "
            f"val_mae={mae:.2f} | "
            f"val_rmse={rmse:.2f}"
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

            print(f"Saved checkpoint: {ckpt_name}")

        else:

            patience_counter += 1

            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break

    # -------------------------
    # Step 10
    # -------------------------
    print("Step 10: Saving history")

    history_path = (
        args.checkpoints_dir
        / f"{args.model_type}_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print("Training completed successfully!")
if __name__ == "__main__":
    main()
