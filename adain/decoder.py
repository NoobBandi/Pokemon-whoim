"""Decoder network — mirrors VGG19 encoder, trained to reconstruct images."""

import torch.nn as nn


class Decoder(nn.Module):
    """Decoder that mirrors the VGG19 encoder architecture.

    Uses Upsample + Conv2d (avoids checkerboard artifacts from ConvTranspose2d).
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Block 4 (from 512 channels)
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Block 3
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Block 2
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Block 1
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)
