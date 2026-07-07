import torch
import torch.nn as nn
from torchvision import models


def _adapt_resnet_input_conv(resnet: nn.Module) -> None:
    """Replace first conv layer for single-channel grayscale X-rays."""
    old_conv = resnet.conv1
    resnet.conv1 = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )


class CNNBaseline(nn.Module):
    """Model A: ResNet-18 backbone with linear regression head."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        _adapt_resnet_input_conv(self.backbone)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1].conv2
