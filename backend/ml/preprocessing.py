"""Image preprocessing utilities for training."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def letterbox_resize(
    image: NDArray[np.uint8], target_size: int
) -> NDArray[np.uint8]:
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def normalize_image(
    image: NDArray[np.uint8], mean: float, std: float
) -> NDArray[np.float32]:
    normalized = image.astype(np.float32) / 255.0
    return ((normalized - mean) / max(std, 1e-6)).astype(np.float32)


def compute_dataset_statistics(
    data_dir: Path, csv_name: str = "train.csv", sample_size: int = 500
) -> dict:
    """Compute mean/std on a sample of training images."""
    data_dir = Path(data_dir)
    csv_path = data_dir / csv_name
    df = pd.read_csv(csv_path)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    image_dir = data_dir / "train" if (data_dir / "train").exists() else data_dir
    pixels = []
    for _, row in df.iterrows():
        path = image_dir / f"{row['id']}.png"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            pixels.append(img.astype(np.float32).ravel() / 255.0)

    stacked = np.concatenate(pixels)
    stats = {"mean": float(stacked.mean()), "std": float(stacked.std())}
    return stats


def save_normalization_stats(stats: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
