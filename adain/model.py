"""Full AdaIN style transfer model — wires encoder + AdaIN + decoder."""

import torch
import torch.nn as nn

from adain.adain import adain
from adain.decoder import Decoder
from adain.encoder import VGLEncoder


class AdaINNet(nn.Module):
    """Adaptive Instance Normalization style transfer network.

    The encoder is frozen (pretrained VGG19). The decoder is trained.
    """

    def __init__(self):
        super().__init__()
        self.encoder = VGLEncoder()
        self.decoder = Decoder()

    def forward(self, content: torch.Tensor, style: torch.Tensor,
                alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            content: (B, 3, H, W) content image tensor
            style: (B, 3, H, W) style image tensor
            alpha: style interpolation strength (0.0=identity, 1.0=full transfer)

        Returns:
            output: (B, 3, H, W) stylized image
            t: (B, C, H', W') AdaIN-transformed features (for loss computation)
        """
        f_content = self.encoder(content)
        f_style = self.encoder(style)

        t = adain(f_content, f_style)

        # Interpolate between original content features and stylized features
        t = alpha * t + (1 - alpha) * f_content

        output = self.decoder(t)
        return output, t
