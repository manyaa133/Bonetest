"""RSNA Pediatric Bone Age dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

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
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.transform = transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = str(row["id"])
        image_path = self.image_dir / f"{image_id}.png"
        if not image_path.exists():
            image_path = self.image_dir / f"{image_id}.jpg"

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = letterbox_resize(image, self.image_size)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        normalized = normalize_image(image, self.norm_mean, self.norm_std)
        tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        bone_age = torch.tensor(float(row["boneage"]), dtype=torch.float32)
        male = torch.tensor(float(row.get("male", 0)), dtype=torch.float32)

        return {
            "image": tensor,
            "bone_age": bone_age,
            "male": male,
            "id": image_id,
        }


def stratified_split(
    df: pd.DataFrame, val_ratio: float = 0.15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["age_bin"] = pd.cut(
        df["boneage"], bins=[0, 60, 120, 216], labels=["young", "mid", "old"]
    )
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["age_bin"],
    )
    return train_df.drop(columns=["age_bin"]), val_df.drop(columns=["age_bin"])


def create_dataloaders(
    data_dir: Path,
    csv_name: str = "train.csv",
    batch_size: int = 16,
    image_size: int = 512,
    num_workers: int = 4,
    train_transform=None,
    norm_mean: float = 0.4523,
    norm_std: float = 0.2118,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    csv_path = data_dir / csv_name
    if not csv_path.exists():
        csv_path = data_dir / "boneage-training-dataset.csv"

    df = pd.read_csv(csv_path)
    image_subdir = data_dir
    if (data_dir / "train").exists():
        image_subdir = data_dir / "train"

    train_df, val_df = stratified_split(df)

    train_ds = BoneAgeDataset(
        train_df, image_subdir, image_size, train_transform, norm_mean, norm_std
    )
    val_ds = BoneAgeDataset(
        val_df, image_subdir, image_size, None, norm_mean, norm_std
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
    return train_loader, val_loader, train_df, val_df
