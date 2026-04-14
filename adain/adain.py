"""Adaptive Instance Normalization layer."""

import torch


def adain(content_features: torch.Tensor, style_features: torch.Tensor,
          eps: float = 1e-5) -> torch.Tensor:
    """Apply Adaptive Instance Normalization.

    Aligns the channel-wise mean and std of content features to match style features.

    Args:
        content_features: (B, C, H, W) content feature map
        style_features: (B, C, H, W) style feature map
        eps: small value for numerical stability

    Returns:
        Normalized features with style statistics, shape (B, C, H, W)
    """
    mu_c = content_features.mean(dim=[2, 3], keepdim=True)
    sigma_c = content_features.std(dim=[2, 3], keepdim=True) + eps

    mu_s = style_features.mean(dim=[2, 3], keepdim=True)
    sigma_s = style_features.std(dim=[2, 3], keepdim=True) + eps

    return sigma_s * (content_features - mu_c) / sigma_c + mu_s
