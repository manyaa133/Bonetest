"""Image preprocessing utilities for bone age prediction."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def letterbox_resize(image: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Resize while preserving aspect ratio using padding.
    """

    h, w = image.shape[:2]

    scale = min(size / h, size / w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((size, size), dtype=np.uint8)

    top = (size - new_h) // 2
    left = (size - new_w) // 2

    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


def normalize_image(
    image: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Normalize grayscale image.
    """

    image = image.astype(np.float32) / 255.0

    image = (image - mean) / std

    return image


def compute_dataset_statistics(
    data_dir: Path,
    image_size: int = 512,
):
    """
    Compute dataset mean and std.
    """

    data_dir = Path(data_dir)

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
        raise FileNotFoundError("Training CSV not found.")

    df = pd.read_csv(csv_path)

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
        raise FileNotFoundError("Image directory not found.")

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    print("Computing dataset statistics...")

    for image_id in tqdm(df["id"]):

        image_path = image_dir / f"{image_id}.png"

        if not image_path.exists():
            image_path = image_dir / f"{image_id}.jpg"

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            continue

        image = letterbox_resize(
            image,
            image_size,
        )

        image = image.astype(np.float32) / 255.0

        pixel_sum += image.sum()
        pixel_sq_sum += np.square(image).sum()
        total_pixels += image.size

    mean = pixel_sum / total_pixels

    std = np.sqrt(
        (pixel_sq_sum / total_pixels) - (mean ** 2)
    )

    return {
        "mean": float(mean),
        "std": float(std),
    }


def save_normalization_stats(
    stats: dict,
    output_path: Path,
):
    """
    Save normalization statistics to JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            stats,
            f,
            indent=4,
        )


def load_normalization_stats(
    path: Path,
):
    """
    Load normalization statistics.
    """

    with open(path, "r") as f:
        return json.load(f)