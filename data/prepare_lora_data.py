"""Prepare LoRA training data: select Pikachu images, rotate, composite, write captions.

This is a fully NON-semantic concept LoRA: every caption is EMPTY and the LoRA
is meant to be trained UNet-only, so the text encoder learns no language and
there is no trigger word at all. The LoRA becomes an always-on appearance shift
— Pikachu's look lives in the UNet weights; at inference an empty prompt applies
it and structure comes from ControlNet (the host outline).

Outputs (under data/lora_training/):
  image/*.png        # originals + flipped/rotated, composited onto white
  image/*.txt        # one caption per image (empty — no semantics)
  meta_cap.json      # consolidated caption metadata
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


# Cleanest core artworks only: clear eyes, red cheeks, black ear tips, yellow
# body. Costume/hat variants (Libre mask, Phd glasses, *-Cap) are excluded
# because they obscure the core features we want the model to learn.
SOURCE_IMAGES = [
    "025.png",
    "025-Starter.png",
    "025-Cosplay.png",
]

# Original + horizontal flip; Pikachu's face is left-right symmetric so flipping
# is safe and doubles the data. Rotations add pose variety. (No vertical flip —
# an upside-down Pikachu is invalid data.)
ROTATION_ANGLES = [0, -5, 5, -10, 10, -15, 15]
FLIPS = [False, True]

# Empty caption — fully non-semantic. The LoRA trains as an always-on
# appearance shift (no trigger word at all); at inference an empty prompt
# applies it. This matches the goal: turn *every* Pokemon into Pikachu.
CAPTION = ""


def crop_to_content(rgba: Image.Image) -> Image.Image:
    """Crop an RGBA image to its non-transparent bounding box."""
    bbox = rgba.split()[-1].getbbox()
    return rgba.crop(bbox) if bbox else rgba


def fit_on_white(rgba: Image.Image, resolution: int, margin: float) -> Image.Image:
    """Scale to fit within `margin` of a square white canvas, centered, as RGB.

    Fitting *after* rotation guarantees the character is never cropped and
    normalizes every augmentation to the same scale.
    """
    max_side = resolution * margin
    w, h = rgba.size
    scale = max_side / max(w, h)
    resized = rgba.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
    )

    canvas = Image.new("RGBA", (resolution, resolution), (255, 255, 255, 255))
    offset = ((resolution - resized.width) // 2, (resolution - resized.height) // 2)
    canvas.paste(resized, offset, resized)  # use alpha as paste mask
    return canvas.convert("RGB")


def augment_one(
    rgba: Image.Image, flip: bool, angle: float, resolution: int, margin: float
) -> Image.Image:
    """Optional flip, rotate (no cropping), then center on a white canvas."""
    from PIL import ImageOps

    img = ImageOps.mirror(rgba) if flip else rgba
    img = crop_to_content(img)
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    return fit_on_white(rotated, resolution, margin)


def save_pair(
    rgb: Image.Image,
    out_dir: Path,
    stem: str,
    caption: str,
    meta: dict,
) -> None:
    rgb.save(out_dir / f"{stem}.png", "PNG")
    (out_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
    meta[stem] = {"caption": caption}


def prepare(
    source_dir: Path,
    output_dir: Path,
    resolution: int = 512,
    margin: float = 0.85,
) -> dict:
    image_dir = output_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    n_saved = 0

    for name in SOURCE_IMAGES:
        src = source_dir / name
        if not src.exists():
            print(f"  ! missing: {src}")
            continue

        rgba = Image.open(src).convert("RGBA")
        base = src.stem

        for flip in FLIPS:
            for angle in ROTATION_ANGLES:
                aug = augment_one(rgba, flip, angle, resolution, margin)
                tag = "f" if flip else "o"
                sign = "p" if angle >= 0 else "m"
                stem = f"{base}_{tag}_r{sign}{abs(angle):02d}"
                save_pair(aug, image_dir, stem, CAPTION, meta)
                n_saved += 1

    meta_path = output_dir / "meta_cap.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "n_saved": n_saved,
        "image_dir": image_dir,
        "meta_path": meta_path,
    }


def main():
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "dataset" / "images"
    output_dir = project_root / "data" / "lora_training"

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()

    summary = prepare(source_dir, output_dir)

    print()
    print(f"Saved:  {summary['n_saved']} images + captions")
    print(f"Images: {summary['image_dir']}")
    print(f"Meta:   {summary['meta_path']}")


if __name__ == "__main__":
    main()
