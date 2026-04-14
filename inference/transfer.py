"""Single and batch style transfer inference."""

import os

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from adain.model import AdaINNet
from data.dataset import PokemonImageDataset
from data.preprocessing import build_transform, denormalize
from data.style_utils import load_style_image
from inference.postprocess import restore_alpha
from utils.config import Config
from utils.device import get_device


def load_model(checkpoint_path: str, device: torch.device) -> AdaINNet:
    """Load a trained AdaIN model from checkpoint."""
    model = AdaINNet()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    model.to(device)
    model.eval()
    return model


def transfer_single(model: AdaINNet, content_tensor: torch.Tensor,
                    style_tensor: torch.Tensor, device: torch.device,
                    alpha: float = 1.0) -> torch.Tensor:
    """Apply AdaIN style transfer to a single content image."""
    model.eval()
    with torch.no_grad():
        content = content_tensor.unsqueeze(0).to(device)
        style = style_tensor.to(device)

        output, _ = model(content, style, alpha=alpha)

        output = denormalize(output)
        output = output.clamp(0, 1)

    return output.squeeze(0).cpu()


def batch_transfer(config: Config | None = None, checkpoint_path: str | None = None,
                   alpha: float | None = None):
    """Apply Pikachu style to all Pokemon images in the dataset."""
    if config is None:
        config = Config()
    if checkpoint_path is None:
        # Use the latest checkpoint
        ckpts = sorted(config.checkpoint_dir.glob("decoder_epoch_*.pth"))
        if not ckpts:
            raise FileNotFoundError("No checkpoints found. Run training first.")
        checkpoint_path = str(ckpts[-1])
    if alpha is None:
        alpha = config.alpha

    device = get_device()

    # Load model
    model = load_model(checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Load style image (Pikachu)
    style_tensor = load_style_image(
        str(config.style_image),
        target_size=config.image_size,
        imagenet_mean=config.imagenet_mean,
        imagenet_std=config.imagenet_std,
    ).to(device)

    # Load dataset
    transform = build_transform(config.image_size, config.imagenet_mean, config.imagenet_std)
    dataset = PokemonImageDataset(
        image_dir=str(config.image_dir),
        transform=transform,
        target_size=config.image_size,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\nRunning batch style transfer on {len(dataset)} images")
    print(f"  Style: {config.style_image.name}")
    print(f"  Alpha: {alpha}")
    print(f"  Output: {config.output_dir}\n")

    for i in tqdm(range(len(dataset)), desc="Transferring style"):
        content_tensor, filename = dataset[i]
        filename_base = os.path.splitext(filename)[0]

        # Transfer style
        result_tensor = transfer_single(model, content_tensor, style_tensor, device, alpha)

        # Convert to PIL
        result_pil = transforms.functional.to_pil_image(result_tensor)

        # Restore original alpha mask
        alpha_mask = dataset.get_alpha(filename)
        if alpha_mask is not None:
            result_pil = restore_alpha(result_pil, alpha_mask)

        # Save
        output_path = config.output_dir / f"{filename_base}_pikachu.png"
        result_pil.save(output_path, "PNG")

    print(f"\nDone! {len(dataset)} stylized images saved to {config.output_dir}")
