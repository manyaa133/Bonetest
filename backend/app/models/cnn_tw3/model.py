from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class CNNTW3(nn.Module):
    """TW3-inspired hybrid model using CNN features plus a proxy skeletal feature branch."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        _adapt_resnet_input_conv(backbone)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.tw3_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(self.feature_dim + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def _tw3_inspired_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        h, w = x.shape[-2], x.shape[-1]
        center_crop = x[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_focus = center_crop.mean(dim=(1, 2, 3), keepdim=True)
        edge_density = torch.mean(
            torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]),
            dim=(1, 2, 3),
            keepdim=True,
        )
        edge_density = edge_density + torch.mean(
            torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]),
            dim=(1, 2, 3),
            keepdim=True,
        )
        edge_density = edge_density / 2.0
        return torch.cat([mean, std, edge_density, center_focus], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_features = self.image_backbone(x)
        if image_features.dim() > 2:
            image_features = torch.flatten(image_features, 1)

        tw3_features = self._tw3_inspired_features(x)
        tw3_features = self.tw3_encoder(tw3_features.squeeze(-1).squeeze(-1))
        fused = torch.cat([image_features, tw3_features], dim=1)
        return self.fusion_head(fused).squeeze(-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.image_backbone.layer4[-1].conv2
