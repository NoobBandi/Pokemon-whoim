"""Style image loader — loads Pikachu 025.png as the style reference."""

import torch
from PIL import Image
from torchvision import transforms

from data.preprocessing import rgba_to_rgb_white_bg


def load_style_image(path: str, target_size: int = 512,
                     imagenet_mean: tuple = (0.485, 0.456, 0.406),
                     imagenet_std: tuple = (0.229, 0.224, 0.225)) -> torch.Tensor:
    """Load and preprocess a style reference image. Returns (1, 3, H, W) tensor."""
    img = Image.open(path).convert("RGBA")
    rgb = rgba_to_rgb_white_bg(img)

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return transform(rgb).unsqueeze(0)  # (1, 3, H, W)
