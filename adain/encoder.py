"""VGG19 encoder — frozen, pretrained, up to relu4_1."""

import torch
import torch.nn as nn
from torchvision import models


# VGG19 feature indices for relu outputs used in style loss
ENCODER_RELUS = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
}
# We slice features up to index 22 (through relu4_1 + conv4_2 region)
# The encoder output used for AdaIN is at relu4_1 (index 20)
ENCODER_SLICE = 22


class VGLEncoder(nn.Module):
    """Pretrained VGG19 encoder, frozen, with max-pool replaced by avg-pool."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(models.VGG19_Weights.DEFAULT).features

        # Replace max-pool with avg-pool for smoother style transfer
        layers = []
        for layer in vgg[:ENCODER_SLICE]:
            if isinstance(layer, nn.MaxPool2d):
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers.append(layer)
        self.layers = nn.Sequential(*layers)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def forward_multi(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Single forward pass returning features at multiple relu layers (for style loss)."""
        features = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in ENCODER_RELUS.values():
                for name, idx in ENCODER_RELUS.items():
                    if idx == i:
                        features[name] = x
        return features
