from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class MultiscaleCNNFeatureExtractor(nn.Module):
    """Feature extractor that concatenates early, mid, and deep CNN features."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        _adapt_resnet_input_conv(backbone)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.early_proj = nn.Linear(64, 128)
        self.mid_proj = nn.Linear(128, 128)
        self.deep_proj = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x0 = self.backbone.maxpool(x0)

        x1 = self.backbone.layer1(x0)
        x2 = self.backbone.layer2(x1)
        x4 = self.backbone.layer4(self.backbone.layer3(x2))

        early_features = self.backbone.avgpool(x1)
        early_features = torch.flatten(early_features, 1)
        mid_features = self.backbone.avgpool(x2)
        mid_features = torch.flatten(mid_features, 1)
        deep_features = self.backbone.avgpool(x4)
        deep_features = torch.flatten(deep_features, 1)

        early_features = self.early_proj(early_features)
        mid_features = self.mid_proj(mid_features)
        deep_features = self.deep_proj(deep_features)
        return torch.cat([early_features, mid_features, deep_features], dim=1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1].conv2


class MultiscaleCNNRFWrapper:
    """Wrapper for multiscale CNN features + sklearn Random Forest regressor."""

    def __init__(
        self,
        cnn: MultiscaleCNNFeatureExtractor | None = None,
        rf_model=None,
        pretrained: bool = False,
    ) -> None:
        self.cnn = cnn or MultiscaleCNNFeatureExtractor(pretrained=pretrained)
        self.rf_model = rf_model

    def predict_numpy(self, features) -> float:
        return float(self.rf_model.predict(features.reshape(1, -1))[0])
