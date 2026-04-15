"""LAB color space transfer — Pikachu color to all Pokemon."""

import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.preprocessing import rgba_to_rgb_white_bg, extract_alpha
from utils.config import Config


def lab_color_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    """Transfer color from source to target in LAB color space.

    Args:
        source_rgb: Source image (Pikachu) in RGB, shape (H, W, 3), dtype uint8
        target_rgb: Target image to recolor, shape (H, W, 3), dtype uint8

    Returns:
        Recolored target image in RGB, shape (H, W, 3), dtype uint8
    """
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    result = np.zeros_like(target_lab)
    for i in range(3):  # L, A, B channels
        s_mean, s_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
        t_mean, t_std = target_lab[:, :, i].mean(), target_lab[:, :, i].std()

        # Avoid division by zero
        if t_std < 1e-6:
            result[:, :, i] = source_lab[:, :, i]
        else:
            result[:, :, i] = (target_lab[:, :, i] - t_mean) * (s_std / t_std) + s_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


def batch_lab_transfer(config: Config | None = None, output_subdir: str = "lab_stylized"):
    """Apply Pikachu color transfer to all Pokemon images via LAB space."""
    if config is None:
        config = Config()

    # Load Pikachu as source
    pikachu_rgba = Image.open(config.style_image).convert("RGBA")
    bg = Image.new("RGBA", pikachu_rgba.size, (255, 255, 255, 255))
    pikachu_rgb = Image.alpha_composite(bg, pikachu_rgba).convert("RGB")
    pikachu_np = np.array(pikachu_rgb)

    image_dir = config.image_dir
    output_dir = config.project_root / "output" / output_subdir
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png") and os.path.isfile(os.path.join(image_dir, f))
    ])

    print(f"LAB color transfer: {len(image_files)} images")
    print(f"  Source (Pikachu): {config.style_image.name}")
    print(f"  Output: {output_dir}\n")

    for filename in tqdm(image_files, desc="LAB transfer"):
        path = os.path.join(image_dir, filename)

        img = Image.open(path).convert("RGBA")
        alpha = extract_alpha(img)

        rgb = rgba_to_rgb_white_bg(img)
        target_np = np.array(rgb)

        result_np = lab_color_transfer(pikachu_np, target_np)
        result_pil = Image.fromarray(result_np)

        # Restore alpha mask
        alpha_resized = alpha.resize(result_pil.size, Image.BILINEAR)
        result_pil = result_pil.convert("RGBA")
        result_pil.putalpha(alpha_resized)

        base = os.path.splitext(filename)[0]
        result_pil.save(output_dir / f"{base}_pikachu_lab.png", "PNG")

    print(f"\nDone! {len(image_files)} images saved to {output_dir}")
