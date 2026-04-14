"""Content loss and style loss (Gram matrix) for AdaIN training."""

import torch
import torch.nn as nn

from adain.encoder import VGLEncoder


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of a feature map.

    Args:
        features: (B, C, H, W)

    Returns:
        Gram matrix: (B, C, C)
    """
    b, c, h, w = features.size()
    f = features.view(b, c, h * w)
    gram = torch.bmm(f, f.transpose(1, 2))
    return gram / (c * h * w)


class ContentLoss(nn.Module):
    """MSE loss between generated features and AdaIN target at relu4_1."""

    def __init__(self, encoder: VGLEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_gen = self.encoder(generated)
        return nn.functional.mse_loss(f_gen, target)


class StyleLoss(nn.Module):
    """MSE loss of Gram matrices at multiple relu layers."""

    def __init__(self, encoder: VGLEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, generated: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Clamp generated output to valid range to prevent NaN in VGG features
        generated = generated.clamp(-10, 10)
        gen_features = self.encoder.forward_multi(generated)
        style_features = self.encoder.forward_multi(style)

        loss = torch.tensor(0.0, device=generated.device)
        for key in gen_features:
            gram_gen = gram_matrix(gen_features[key])
            gram_style = gram_matrix(style_features[key])
            loss = loss + nn.functional.mse_loss(gram_gen, gram_style)

        return loss
