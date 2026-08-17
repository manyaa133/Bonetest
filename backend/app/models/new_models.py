"""Additional bone age regression models.

The RSNA dataset does not provide segmentation masks, bounding boxes, or TW3
stage labels. The implementations below therefore use scientifically defensible
proxy branches that are compatible with the existing training and evaluation
pipeline while clearly documenting the limitation.
"""

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
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

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
