import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class CNNFeatureExtractor(nn.Module):
    """CNN backbone that outputs 512-d features for Random Forest regression."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        _adapt_resnet_input_conv(backbone)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return features

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1].conv2


class CNNWithRFWrapper:
    """Model D: PyTorch feature extractor + sklearn Random Forest."""

    def __init__(self, cnn: CNNFeatureExtractor, rf_model) -> None:
        self.cnn = cnn
        self.rf_model = rf_model

    def predict_numpy(self, features) -> float:
        return float(self.rf_model.predict(features.reshape(1, -1))[0])
