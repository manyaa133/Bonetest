from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class EnsembleCNN(nn.Module):
    """Multi-branch ensemble of CNN regressors using learned fusion."""

    def __init__(self, pretrained: bool = False, num_branches: int = 3) -> None:
        super().__init__()
        if num_branches < 2:
            raise ValueError("num_branches must be at least 2")

        self.num_branches = num_branches
        self.branches = nn.ModuleList(
            [self._build_branch(pretrained=pretrained) for _ in range(num_branches)]
        )
        self.ensemble_head = nn.Sequential(
            nn.Linear(num_branches, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def _build_branch(self, pretrained: bool) -> nn.Module:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        _adapt_resnet_input_conv(backbone)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = [branch(x).squeeze(-1) for branch in self.branches]
        ensemble_input = torch.stack(predictions, dim=1)
        return self.ensemble_head(ensemble_input).squeeze(-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.branches[0].layer4[-1].conv2
