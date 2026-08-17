from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class RegressionMultimodal(nn.Module):
    """Regression-based multimodal model using image and gender metadata."""

    def __init__(self, pretrained: bool = False, meta_dim: int = 1) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        _adapt_resnet_input_conv(backbone)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(self.feature_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def _prepare_gender(self, gender: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if gender is None:
            gender = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
        elif gender.dim() == 1:
            gender = gender.unsqueeze(-1)
        return gender.float()

    def forward(self, x: torch.Tensor, gender: torch.Tensor | None = None) -> torch.Tensor:
        image_features = self.image_backbone(x)
        if image_features.dim() > 2:
            image_features = torch.flatten(image_features, 1)

        gender_tensor = self._prepare_gender(gender, x.size(0), x.device)
        meta_features = self.meta_encoder(gender_tensor)
        fused = torch.cat([image_features, meta_features], dim=1)
        return self.fusion_head(fused).squeeze(-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.image_backbone.layer4[-1].conv2
