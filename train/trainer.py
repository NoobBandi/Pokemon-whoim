"""Training loop for AdaIN decoder."""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from adain.model import AdaINNet
from data.dataset import PokemonImageDataset
from data.preprocessing import build_transform
from train.losses import ContentLoss, StyleLoss
from utils.config import Config
from utils.device import get_device


def train(config: Config | None = None):
    """Train the AdaIN decoder on Pokemon images."""
    if config is None:
        config = Config()

    device = get_device()
    is_cuda = device.type == "cuda"

    # Build dataset and dataloader
    transform = build_transform(config.image_size, config.imagenet_mean, config.imagenet_std)
    dataset = PokemonImageDataset(
        image_dir=str(config.image_dir),
        transform=transform,
        target_size=config.image_size,
    )

    # Use adjusted batch size for MPS
    batch_size = config.batch_size
    if device.type == "mps":
        batch_size = min(batch_size, 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if is_cuda else 0,
        pin_memory=is_cuda,
        drop_last=True,
    )

    # Build model
    model = AdaINNet().to(device)
    # Only train the decoder
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config.learning_rate)

    # Loss functions share the same encoder as the model
    content_loss_fn = ContentLoss(model.encoder)
    style_loss_fn = StyleLoss(model.encoder)

    # Mixed precision disabled — causes NaN with small decoder outputs
    scaler = None

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\nTraining AdaIN decoder")
    print(f"  Dataset: {len(dataset)} images")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Style: {config.style_image.name} (fixed)")
    print(f"  Device: {device}\n")

    # Load fixed Pikachu style tensor once
    from data.style_utils import load_style_image
    style_fixed = load_style_image(
        str(config.style_image),
        target_size=config.image_size,
        imagenet_mean=config.imagenet_mean,
        imagenet_std=config.imagenet_std,
    ).to(device)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_content_loss = 0.0
        epoch_style_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch_idx, (content_batch, _) in enumerate(pbar):
            content_batch = content_batch.to(device)

            # Fixed Pikachu style for every batch
            style_batch = style_fixed.expand(content_batch.size(0), -1, -1, -1)

            optimizer.zero_grad()

            output, t = model(content_batch, style_batch)
            c_loss = content_loss_fn(output, t)
            s_loss = style_loss_fn(output, style_batch)
            total_loss = config.content_weight * c_loss + config.style_weight * s_loss

            total_loss.backward()
            optimizer.step()

            epoch_content_loss += c_loss.item()
            epoch_style_loss += s_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "c_loss": f"{c_loss.item():.4f}",
                "s_loss": f"{s_loss.item():.4f}",
            })

        avg_c = epoch_content_loss / num_batches
        avg_s = epoch_style_loss / num_batches
        print(f"Epoch {epoch + 1} — Content: {avg_c:.4f}, Style: {avg_s:.4f}")

        # Save checkpoint
        ckpt_path = config.checkpoint_dir / f"decoder_epoch_{epoch + 1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "decoder_state_dict": model.decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "content_loss": avg_c,
            "style_loss": avg_s,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    print("\nTraining complete!")
    return model
