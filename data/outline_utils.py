"""Outline / silhouette extraction for Pokemon images.

For HybridShivam PNGs the alpha channel already IS a pixel-perfect mask, so no
model is needed — extraction is instant. For images without an alpha channel
(other datasets, photos) we fall back to rembg (U2Net) background removal.

Three outline styles:
    silhouette  — solid black shape on white (the "Who's that Pokemon?" look)
    contour     — outer outline stroke only, on white
    mask        — raw white-on-black alpha mask

CLI:
    python -m data.outline_utils --input 006.png --style silhouette
    python -m data.outline_utils --batch --style contour
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Alpha values at/below this are treated as background.
_ALPHA_THRESHOLD = 16


def get_mask(pil_image: Image.Image) -> Image.Image:
    """Return a white-on-black foreground mask (L mode) for an image.

    Uses the alpha channel when present (instant, exact). Otherwise falls back
    to rembg/U2Net background removal.
    """
    if pil_image.mode == "RGBA":
        alpha = np.array(pil_image.split()[-1])
        mask = np.where(alpha > _ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
        return Image.fromarray(mask, mode="L")

    return _mask_via_rembg(pil_image)


def _mask_via_rembg(pil_image: Image.Image) -> Image.Image:
    """Background removal for images that have no alpha channel."""
    try:
        from rembg import remove
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "Image has no alpha channel and rembg is not installed. "
            "Install it with `pip install rembg` to extract outlines from "
            "non-transparent images."
        ) from exc

    cutout = remove(pil_image.convert("RGBA"))
    alpha = np.array(cutout.split()[-1])
    mask = np.where(alpha > _ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def to_silhouette(mask: Image.Image) -> Image.Image:
    """Solid black foreground shape on a white background (RGB)."""
    m = np.array(mask)
    rgb = np.full((*m.shape, 3), 255, dtype=np.uint8)
    rgb[m > 127] = (0, 0, 0)
    return Image.fromarray(rgb, mode="RGB")


def to_contour(mask: Image.Image, thickness: int = 3) -> Image.Image:
    """Outer outline stroke only (black on white, RGB)."""
    import cv2  # only the contour style needs OpenCV

    m = np.array(mask)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.full((*m.shape, 3), 255, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, (0, 0, 0), thickness)
    return Image.fromarray(canvas, mode="RGB")


def to_lineart_canny(
    pil_image: Image.Image,
    low_threshold: int = 80,
    high_threshold: int = 160,
) -> Image.Image:
    """Line art via Canny: outer outline + internal body lines (black on white).

    Composites onto white first so the alpha edge does not create a false
    silhouette border, then inverts Canny (black lines on white) to match the
    clean line-drawing look.
    """
    import cv2

    if pil_image.mode == "RGBA":
        rgb = Image.alpha_composite(
            Image.new("RGBA", pil_image.size, (255, 255, 255, 255)), pil_image
        ).convert("RGB")
    else:
        rgb = pil_image.convert("RGB")

    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    lines = 255 - edges  # black lines on white
    return Image.fromarray(lines, mode="L").convert("RGB")


def extract_outline(
    pil_image: Image.Image,
    style: str = "silhouette",
    contour_thickness: int = 3,
) -> Image.Image:
    """Extract an outline from an image in the requested style."""
    if style == "canny":
        return to_lineart_canny(pil_image)

    mask = get_mask(pil_image)

    if style == "silhouette":
        return to_silhouette(mask)
    if style == "contour":
        return to_contour(mask, thickness=contour_thickness)
    if style == "mask":
        return mask.convert("RGB")

    raise ValueError(f"Unknown style: {style!r} (use silhouette|contour|mask|canny)")


def _resolve_image_dir() -> Path:
    """Locate the default Pokemon image directory."""
    root = Path(__file__).resolve().parent.parent
    return root / "dataset" / "images"


def _process_file(
    image_dir: Path,
    filename: str,
    output_dir: Path,
    style: str,
    thickness: int,
) -> None:
    img = Image.open(image_dir / filename)
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    outline = extract_outline(img, style=style, contour_thickness=thickness)
    base = os.path.splitext(filename)[0]
    out_path = output_dir / f"{base}_{style}.png"
    outline.save(out_path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Pokemon outlines/silhouettes.")
    parser.add_argument("--input", help="Single image filename (e.g. 006.png)")
    parser.add_argument("--batch", action="store_true", help="Process all PNGs in the image dir")
    parser.add_argument(
        "--style",
        choices=["silhouette", "contour", "mask", "canny"],
        default="silhouette",
        help="Outline style",
    )
    parser.add_argument("--thickness", type=int, default=3, help="Contour stroke thickness")
    parser.add_argument("--image-dir", help="Override image directory")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    image_dir = Path(args.image_dir) if args.image_dir else _resolve_image_dir()
    root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else root / "output" / "outlines"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        filename = args.input if args.input.lower().endswith(".png") else args.input + ".png"
        if not (image_dir / filename).exists():
            parser.error(f"Image '{filename}' not found in {image_dir}")
        _process_file(image_dir, filename, output_dir, args.style, args.thickness)
        print(f"Saved: {output_dir}/{os.path.splitext(filename)[0]}_{args.style}.png")

    elif args.batch:
        files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".png"))
        print(f"Extracting '{args.style}' outlines for {len(files)} images -> {output_dir}")
        for filename in files:
            try:
                _process_file(image_dir, filename, output_dir, args.style, args.thickness)
            except Exception as exc:  # noqa: BLE001
                print(f"  Failed {filename}: {exc}")
        print("Done.")

    else:
        parser.error("Provide --input <file> or --batch")


if __name__ == "__main__":
    main()
