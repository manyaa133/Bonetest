"""Training-time image augmentation."""

from __future__ import annotations

import cv2
import numpy as np


class TrainAugmentation:
    """OpenCV-based augmentation pipeline for bone age X-rays."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_limit: int = 10,
        brightness_limit: float = 0.15,
        noise_std: float = 0.01,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotate_limit = rotate_limit
        self.brightness_limit = brightness_limit
        self.noise_std = noise_std

    def __call__(self, image: np.ndarray) -> dict:
        img = image.copy()

        if np.random.random() < self.flip_prob:
            img = cv2.flip(img, 1)

        angle = np.random.uniform(-self.rotate_limit, self.rotate_limit)
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        factor = 1.0 + np.random.uniform(
            -self.brightness_limit, self.brightness_limit
        )
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std * 255, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        scale = np.random.uniform(0.9, 1.0)
        if scale < 1.0:
            nh, nw = int(img.shape[0] * scale), int(img.shape[1] * scale)
            cropped = cv2.resize(img, (nw, nh))
            canvas = np.zeros_like(img)
            y0 = (img.shape[0] - nh) // 2
            x0 = (img.shape[1] - nw) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = cropped
            img = canvas

        return {"image": img}
