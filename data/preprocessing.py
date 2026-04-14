"""RGBA-to-RGB compositing, resize, and ImageNet normalization."""

from PIL import Image
import torch
from torchvision import transforms


def rgba_to_rgb_white_bg(pil_image: Image.Image) -> Image.Image:
    """Composite RGBA image onto white background, return RGB."""
    background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(background, pil_image)
    return composite.convert("RGB")


def extract_alpha(pil_image: Image.Image) -> Image.Image:
    """Extract the alpha channel from an RGBA image."""
    return pil_image.split()[-1].copy()


def build_transform(image_size: int = 512,
                    imagenet_mean: tuple = (0.485, 0.456, 0.406),
                    imagenet_std: tuple = (0.229, 0.224, 0.225)) -> transforms.Compose:
    """Build the standard preprocessing transform pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


def denormalize(tensor: torch.Tensor,
                imagenet_mean: tuple = (0.485, 0.456, 0.406),
                imagenet_std: tuple = (0.229, 0.224, 0.225)) -> torch.Tensor:
    """Reverse ImageNet normalization to get [0,1] pixel values."""
    mean = torch.tensor(imagenet_mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(imagenet_std).view(-1, 1, 1).to(tensor.device)
    return tensor * std + mean
