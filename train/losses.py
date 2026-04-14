"""Content loss and style loss for AdaIN training."""

import torch
import torch.nn as nn

from adain.encoder import VGLEncoder


def calc_mean_std(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-wise mean and std of feature maps."""
    b, c, h, w = features.size()
    features_view = features.view(b, c, -1)
    mean = features_view.mean(dim=2).view(b, c, 1, 1)
    std = features_view.std(dim=2).view(b, c, 1, 1) + 1e-5
    return mean, std


class ContentLoss(nn.Module):
    """MSE between encoder output of generated image and AdaIN target."""

    def __init__(self, encoder: VGLEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_gen = self.encoder(generated)
        return nn.functional.mse_loss(f_gen, target)


class StyleLoss(nn.Module):
    """MSE of mean and std at multiple relu layers (AdaIN paper definition)."""

    def __init__(self, encoder: VGLEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, generated: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gen_features = self.encoder.forward_multi(generated)
        style_features = self.encoder.forward_multi(style)

        loss = torch.tensor(0.0, device=generated.device)
        for key in gen_features:
            mu_gen, sigma_gen = calc_mean_std(gen_features[key])
            mu_style, sigma_style = calc_mean_std(style_features[key])

            loss = loss + nn.functional.mse_loss(mu_gen, mu_style)
            loss = loss + nn.functional.mse_loss(sigma_gen, sigma_style)

        return loss
