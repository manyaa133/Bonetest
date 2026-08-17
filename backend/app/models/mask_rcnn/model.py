from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class MaskRCNNRegression(nn.Module):
    """Proxy Mask R-CNN-style region guided regressor.

    The RSNA dataset does not provide instance segmentation masks or bounding
    boxes, so this module uses a lightweight proposal head with a pseudo-mask
    to focus the CNN features on the most informative image region.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        _adapt_resnet_input_conv(backbone)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.mask_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def _extract_feature_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.image_backbone.conv1(x)
        x0 = self.image_backbone.bn1(x0)
        x0 = self.image_backbone.relu(x0)
        x0 = self.image_backbone.maxpool(x0)

        x1 = self.image_backbone.layer1(x0)
        x2 = self.image_backbone.layer2(x1)
        x3 = self.image_backbone.layer3(x2)
        x4 = self.image_backbone.layer4(x3)

        proposal_logits = self.mask_head(x4)
        proposal_mask = torch.sigmoid(proposal_logits).squeeze(1)
        pooled_features = torch.mean(x4 * proposal_mask.unsqueeze(1), dim=(2, 3))
        return pooled_features, proposal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.image_backbone(x)
        if feature_map.dim() > 2:
            feature_map = torch.flatten(feature_map, 1)

        proposal_features, _ = self._extract_feature_map(x)
        fused = torch.cat([feature_map, proposal_features], dim=1)
        return self.regression_head(fused).squeeze(-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.image_backbone.layer4[-1].conv2
