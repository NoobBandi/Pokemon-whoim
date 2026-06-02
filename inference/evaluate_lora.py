"""Evaluate Pikachu LoRA checkpoints with ControlNet on real Pokemon outlines.

This is the real checkpoint selector. The free training samples only show whether
the LoRA learned Pikachu at all; this runs the actual non-semantic pipeline —
empty prompt, structure from ControlNet (Canny of the host Pokemon), Pikachu
appearance from the LoRA — and builds a comparison grid:

    rows = [Canny] + one row per checkpoint
    cols = each test Pokemon

Run on the training box (GPU + SD already cached). The .venv needs opencv:
    pip install opencv-python

    python -m inference.evaluate_lora
    python -m inference.evaluate_lora --lora-scale 0.9 --controlnet-scale 0.9
    python -m inference.evaluate_lora --checkpoints output/lora/pikachu_lora_v1-step00001500.safetensors

No text, no semantics: prompt is always "".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from data.outline_utils import controlnet_canny
from data.preprocessing import rgba_to_rgb_white_bg
from utils.config import Config
from utils.device import get_device


def build_pipeline(config: Config, device: torch.device, dtype: torch.dtype):
    """SD 1.5 + ControlNet Canny, no IP-Adapter (LoRA carries the appearance)."""
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    controlnet = ControlNetModel.from_pretrained(config.controlnet_model_id, torch_dtype=dtype)
    vae = (
        AutoencoderKL.from_pretrained(config.vae_model_id, torch_dtype=dtype)
        if config.vae_model_id
        else None
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.sd_model_id,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    if device.type == "mps":
        pipe.enable_attention_slicing()
    return pipe


def canny_of(path: Path, config: Config) -> tuple[Image.Image, Image.Image]:
    """Return (white-bg RGB, Canny edge map) at the model resolution."""
    rgb = rgba_to_rgb_white_bg(Image.open(path).convert("RGBA"))
    rgb = rgb.resize((config.image_size, config.image_size), Image.LANCZOS)
    canny = controlnet_canny(
        rgb,
        low_threshold=config.canny_low_threshold,
        high_threshold=config.canny_high_threshold,
    )
    return rgb, canny


def _label(img: Image.Image, text: str) -> None:
    """Draw a small caption in the top-left corner (in place)."""
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))


def _generate(pipe, canny, lora_scale, cnet_scale, args):
    """One generation: prompt (default empty) + ControlNet + LoRA."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    return pipe(
        prompt=args.prompt,
        image=canny,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=cnet_scale,
        cross_attention_kwargs={"scale": lora_scale},
        generator=generator,
    ).images[0]


def run_sweep(pipe, config, args, root):
    """Sweep lora_scale (rows) x controlnet_scale (cols) for one checkpoint/input."""
    image_dir = Path(args.image_dir)
    ckpt = Path(args.checkpoint) if args.checkpoint else (
        root / "output" / "lora" / "pikachu_lora_v1.safetensors"
    )
    _rgb, canny = canny_of(image_dir / args.sweep_input, config)

    lora_scales = args.lora_scales
    cnet_scales = args.controlnet_scales
    print(f"Sweep on {ckpt.name}, input {args.sweep_input}")
    print(f"  lora_scales (rows): {lora_scales}")
    print(f"  controlnet_scales (cols): {cnet_scales}")

    pipe.load_lora_weights(str(ckpt))
    cell = config.image_size
    grid = Image.new("RGB", (cell * len(cnet_scales), cell * len(lora_scales)), (30, 30, 30))
    for i, ls in enumerate(lora_scales):
        for j, cs in enumerate(cnet_scales):
            out = _generate(pipe, canny, ls, cs, args).resize((cell, cell))
            _label(out, f"L{ls}/C{cs}")
            grid.paste(out, (j * cell, i * cell))
        print(f"  lora_scale={ls} done")
    pipe.unload_lora_weights()

    out_path = Path(args.output).with_name("sweep.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, "PNG")
    print(f"\nSaved sweep grid: {out_path}")
    print("Rows = lora_scale (down = stronger Pikachu); Cols = controlnet_scale")
    print("(right = stronger host structure). Pick the cell that keeps the outline")
    print("AND shows yellow + red cheeks + black ear tips.")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate Pikachu LoRA checkpoints via ControlNet.")
    parser.add_argument("--image-dir", default=str(root / "inference" / "eval_inputs"))
    parser.add_argument("--inputs", nargs="*", help="Filenames in image-dir (default: all PNGs)")
    parser.add_argument("--checkpoints", nargs="*", help="LoRA .safetensors (default: all in output/lora)")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--controlnet-scale", type=float, default=0.9,
                        help="High = host structure priority")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--prompt", default="",
                        help="Empty = pure non-semantic (CFG no-op). A short prompt re-enables CFG.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(root / "output" / "lora_eval" / "grid.png"))
    # Sweep mode: find the lora/controlnet balance on one checkpoint + one input.
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep lora_scale x controlnet_scale instead of comparing checkpoints")
    parser.add_argument("--checkpoint", help="Single checkpoint for sweep (default: final)")
    parser.add_argument("--sweep-input", default="004.png", help="Single input for sweep")
    parser.add_argument("--lora-scales", nargs="*", type=float,
                        default=[0.8, 1.0, 1.2, 1.4], help="Sweep rows")
    parser.add_argument("--controlnet-scales", nargs="*", type=float,
                        default=[0.4, 0.55, 0.7, 0.85], help="Sweep cols")
    args = parser.parse_args()

    config = Config()
    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if args.sweep:
        pipe = build_pipeline(config, device, dtype)
        run_sweep(pipe, config, args, root)
        return

    image_dir = Path(args.image_dir)
    inputs = args.inputs or sorted(f.name for f in image_dir.glob("*.png"))
    if not inputs:
        parser.error(f"No input images in {image_dir}")

    ckpt_dir = root / "output" / "lora"
    checkpoints = [Path(c) for c in args.checkpoints] if args.checkpoints else sorted(
        ckpt_dir.glob("*.safetensors")
    )
    if not checkpoints:
        parser.error(f"No LoRA checkpoints in {ckpt_dir}")

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Inputs ({len(inputs)}): {inputs}")
    print(f"Checkpoints ({len(checkpoints)}): {[c.name for c in checkpoints]}")
    print(f"lora_scale={args.lora_scale} controlnet_scale={args.controlnet_scale}")

    pipe = build_pipeline(config, device, dtype)

    # Precompute Canny per input.
    cols = [(name, *canny_of(image_dir / name, config)) for name in inputs]

    cell = config.image_size
    n_rows = len(checkpoints) + 1  # header row of Canny maps
    grid = Image.new("RGB", (cell * len(cols), cell * n_rows), (30, 30, 30))

    for j, (name, _rgb, canny) in enumerate(cols):
        header = canny.resize((cell, cell)).convert("RGB")
        _label(header, name)
        grid.paste(header, (j * cell, 0))

    for i, ckpt in enumerate(checkpoints):
        step_tag = ckpt.stem.split("step")[-1].lstrip("0") or "final"
        print(f"[{i + 1}/{len(checkpoints)}] {ckpt.name}")
        pipe.load_lora_weights(str(ckpt))
        for j, (name, _rgb, canny) in enumerate(cols):
            out = _generate(pipe, canny, args.lora_scale, args.controlnet_scale, args)
            out = out.resize((cell, cell))
            _label(out, f"s{step_tag}")
            grid.paste(out, (j * cell, (i + 1) * cell))
        pipe.unload_lora_weights()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, "PNG")
    print(f"\nSaved comparison grid: {out_path}")
    print("Rows top->bottom: Canny, then each checkpoint. Pick the row that best")
    print("keeps the host outline AND looks like Pikachu (yellow/red cheeks/black ears).")


if __name__ == "__main__":
    main()
