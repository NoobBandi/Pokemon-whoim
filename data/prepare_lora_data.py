"""Prepare LoRA training data: select Pikachu images, rotate, composite, write captions.

Outputs (under data/lora_training/):
  image/*.png        # originals + rotated, composited onto white
  image/*.txt        # one caption per image (sd-scripts default convention)
  meta_cap.json      # consolidated caption metadata (sd-scripts format)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from data.preprocessing import rgba_to_rgb_white_bg


SOURCE_IMAGES = [
    "025.png",
    "025-Starter.png",
    "025-Libre.png",
    "025-Gmax.png",
    "025-Phd.png",
    "025-Cosplay.png",
]

ROTATION_ANGLES = [-15, -10, -5, 5, 10, 15]

TRIGGER = "pkmn-pikachu"

CAPTION_TEMPLATES = [
    (
        f"{TRIGGER}, a small yellow electric-type pokemon with round red cheek pouches, "
        f"pointed ears with black tips, a zigzag lightning-shaped tail, "
        f"simple cartoon illustration on white background"
    ),
    f"{TRIGGER}, yellow pokemon, white background",
    f"{TRIGGER} standing, yellow fur with red cheeks, cartoon style, white background",
    (
        f"{TRIGGER}, yellow creature with red cheek patches and black ear tips, "
        f"sugimori-style pokemon illustration, white background"
    ),
    f"{TRIGGER}, bright yellow body, brown stripes on back, white background",
]


def alpha_touches_border(rgba: Image.Image, alpha_threshold: int = 10) -> bool:
    """True if any non-transparent pixel sits on the canvas border."""
    alpha = rgba.split()[-1].point(lambda v: 255 if v > alpha_threshold else 0)
    bbox = alpha.getbbox()
    if bbox is None:
        return False
    left, top, right, bottom = bbox
    w, h = alpha.size
    return left == 0 or top == 0 or right == w or bottom == h


def rotate_rgba(rgba: Image.Image, angle: float) -> Image.Image:
    return rgba.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=False,
        fillcolor=(0, 0, 0, 0),
    )


def save_pair(
    rgba: Image.Image,
    out_dir: Path,
    stem: str,
    caption: str,
    meta: dict,
) -> None:
    rgb = rgba_to_rgb_white_bg(rgba)
    rgb.save(out_dir / f"{stem}.png", "PNG")
    (out_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
    meta[stem] = {"caption": caption}


def prepare(source_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    image_dir = output_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    meta: dict[str, dict] = {}
    n_saved = 0
    n_skipped = 0

    for name in SOURCE_IMAGES:
        src = source_dir / name
        if not src.exists():
            print(f"  ! missing: {src}")
            continue

        rgba = Image.open(src).convert("RGBA")
        base = src.stem

        save_pair(rgba, image_dir, base, rng.choice(CAPTION_TEMPLATES), meta)
        n_saved += 1

        for angle in ROTATION_ANGLES:
            rotated = rotate_rgba(rgba, angle)
            if alpha_touches_border(rotated):
                print(f"  - skip {base} angle={angle:+d} (character cropped)")
                n_skipped += 1
                continue
            sign = "p" if angle > 0 else "n"
            stem = f"{base}_rot{sign}{abs(angle):02d}"
            save_pair(rotated, image_dir, stem, rng.choice(CAPTION_TEMPLATES), meta)
            n_saved += 1

    meta_path = output_dir / "meta_cap.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "n_saved": n_saved,
        "n_skipped": n_skipped,
        "image_dir": image_dir,
        "meta_path": meta_path,
    }


def main():
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "dataset" / "HybridShivam-Pokemon" / "assets" / "images"
    output_dir = project_root / "data" / "lora_training"

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()

    summary = prepare(source_dir, output_dir)

    print()
    print(f"Saved:   {summary['n_saved']} images")
    print(f"Skipped: {summary['n_skipped']} cropped rotations")
    print(f"Images:  {summary['image_dir']}")
    print(f"Meta:    {summary['meta_path']}")


if __name__ == "__main__":
    main()
