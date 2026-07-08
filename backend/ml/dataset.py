"""RSNA Pediatric Bone Age dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ml.preprocessing import letterbox_resize, normalize_image


class BoneAgeDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        image_size: int = 512,
        transform: Optional[Callable] = None,
        norm_mean: float = 0.4523,
        norm_std: float = 0.2118,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.transform = transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row["id"])

        image_path = self.image_dir / f"{image_id}.png"
        if not image_path.exists():
            image_path = self.image_dir / f"{image_id}.jpg"

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image missing: {image_path}")

        image = letterbox_resize(image, self.image_size)

        if self.transform:
            image = self.transform(image=image)["image"]

        image = normalize_image(
            image,
            self.norm_mean,
            self.norm_std,
        )

        image = torch.from_numpy(image).unsqueeze(0).float()

        age = torch.tensor(
            float(row["boneage"]),
            dtype=torch.float32,
        )

        male = torch.tensor(
            float(row.get("male", 0)),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "bone_age": age,
            "male": male,
            "id": image_id,
        }


def stratified_split(
    df,
    val_ratio=0.15,
    seed=42,
):
    df = df.copy()

    # Remove rows with missing bone age
    df = df.dropna(subset=["boneage"])

    # Create age bins
    df["age_bin"] = pd.cut(
        df["boneage"],
        bins=[0, 60, 120, 180, 228],
        labels=[
            "young",
            "mid",
            "old",
            "late",
        ],
    )

    # Remove rows that couldn't be assigned to a bin
    df = df.dropna(subset=["age_bin"])

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["age_bin"],
    )

    return (
        train_df.drop(columns=["age_bin"]),
        val_df.drop(columns=["age_bin"]),
    )


def create_dataloaders(
    data_dir: Path,
    csv_name="train.csv",
    batch_size=16,
    image_size=512,
    num_workers=2,
    train_transform=None,
    norm_mean=0.4523,
    norm_std=0.2118,
):
    data_dir = Path(data_dir)

    # Locate CSV
    possible_csv = [
        data_dir / "train.csv",
        data_dir / "boneage-training-dataset.csv",
    ]

    csv_path = None

    for p in possible_csv:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(
            "No training CSV found."
        )

    df = pd.read_csv(csv_path)

    # Locate image directory
    possible_dirs = [
        data_dir / "train",
        data_dir / "boneage-training-dataset" / "boneage-training-dataset",
        data_dir / "boneage-training-dataset",
        data_dir,
    ]

    image_dir = None

    for d in possible_dirs:
        if d.exists():
            image_dir = d
            break

    if image_dir is None:
        raise FileNotFoundError(
            "Image directory not found."
        )

    train_df, val_df = stratified_split(df)

    train_ds = BoneAgeDataset(
        train_df,
        image_dir,
        image_size,
        train_transform,
        norm_mean,
        norm_std,
    )

    val_ds = BoneAgeDataset(
        val_df,
        image_dir,
        image_size,
        None,
        norm_mean,
        norm_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        train_df,
        val_df,
    )