from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class CNNTW3(nn.Module):
    """
    CNN + TW3-inspired hybrid model for bone-age prediction.

    Architecture:
        X-ray
          |
          +--------------------+
          |                    |
          v                    v
      ResNet-18          TW3-inspired
      CNN branch         feature branch
          |                    |
          v                    v
      CNN features       TW3 features
          |                    |
          +---------+----------+
                    |
                    v
                 Fusion
                    |
                    v
             Bone-age output

    The pretrained CNN checkpoint is loaded by train.py.
    The CNN backbone can then be frozen during CNN+TW3 training.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # 1. ResNet-18 CNN backbone
        # ---------------------------------------------------------
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = models.resnet18(
            weights=weights
        )

        # Adapt ResNet-18 for grayscale X-ray images
        _adapt_resnet_input_conv(backbone)

        # Number of features produced by ResNet-18
        self.feature_dim = backbone.fc.in_features

        # Remove the original ImageNet classification layer
        backbone.fc = nn.Identity()

        self.image_backbone = backbone

        # ---------------------------------------------------------
        # 2. TW3-inspired feature branch
        # ---------------------------------------------------------
        #
        # Four proxy skeletal/image features:
        #   1. Mean intensity
        #   2. Standard deviation
        #   3. Edge density
        #   4. Central-region intensity
        #
        self.tw3_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------
        # 3. CNN + TW3 fusion head
        # ---------------------------------------------------------
        self.fusion_head = nn.Sequential(
            nn.Linear(
                self.feature_dim + 32,
                256,
            ),

            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(128, 1),
        )

    # =============================================================
    # TW3-inspired feature extraction
    # =============================================================

    def _tw3_inspired_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract four image-derived TW3-inspired proxy features.

        Returns:
            Tensor of shape [batch_size, 4]
        """

        x = x.float()

        # ---------------------------------------------------------
        # Feature 1: Mean intensity
        # ---------------------------------------------------------
        mean = x.mean(
            dim=(1, 2, 3),
            keepdim=True,
        )

        # ---------------------------------------------------------
        # Feature 2: Standard deviation
        # ---------------------------------------------------------
        std = x.std(
            dim=(1, 2, 3),
            keepdim=True,
        )

        # ---------------------------------------------------------
        # Image dimensions
        # ---------------------------------------------------------
        h = x.shape[-2]
        w = x.shape[-1]

        # ---------------------------------------------------------
        # Central region
        # ---------------------------------------------------------
        center_crop = x[
            :,
            :,
            h // 4 : 3 * h // 4,
            w // 4 : 3 * w // 4,
        ]

        # Feature 4: Central-region intensity
        center_focus = center_crop.mean(
            dim=(1, 2, 3),
            keepdim=True,
        )

        # ---------------------------------------------------------
        # Feature 3: Edge density
        # ---------------------------------------------------------

        vertical_edges = torch.mean(
            torch.abs(
                x[:, :, 1:, :] -
                x[:, :, :-1, :]
            ),
            dim=(1, 2, 3),
            keepdim=True,
        )

        horizontal_edges = torch.mean(
            torch.abs(
                x[:, :, :, 1:] -
                x[:, :, :, :-1]
            ),
            dim=(1, 2, 3),
            keepdim=True,
        )

        edge_density = (
            vertical_edges +
            horizontal_edges
        ) / 2.0

        # ---------------------------------------------------------
        # Combine four features
        # ---------------------------------------------------------

        features = torch.cat(
            [
                mean,
                std,
                edge_density,
                center_focus,
            ],
            dim=1,
        )

        # [B, 4, 1, 1] -> [B, 4]
        features = features.squeeze(-1).squeeze(-1)

        return features

    # =============================================================
    # Forward pass
    # =============================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Grayscale X-ray tensor of shape
                [batch_size, 1, H, W]

        Returns:
            Predicted bone age of shape [batch_size]
        """

        # ---------------------------------------------------------
        # CNN branch
        # ---------------------------------------------------------
        image_features = self.image_backbone(x)

        if image_features.dim() > 2:
            image_features = torch.flatten(
                image_features,
                start_dim=1,
            )

        # ---------------------------------------------------------
        # TW3-inspired branch
        # ---------------------------------------------------------
        tw3_features = self._tw3_inspired_features(x)

        tw3_features = self.tw3_encoder(
            tw3_features
        )

        # ---------------------------------------------------------
        # Feature fusion
        # ---------------------------------------------------------
        fused_features = torch.cat(
            [
                image_features,
                tw3_features,
            ],
            dim=1,
        )

        # ---------------------------------------------------------
        # Bone-age regression
        # ---------------------------------------------------------
        output = self.fusion_head(
            fused_features
        )

        return output.squeeze(-1)

    # =============================================================
    # Grad-CAM support
    # =============================================================

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Return the target convolutional layer for Grad-CAM.
        """

        return self.image_backbone.layer4[-1].conv2