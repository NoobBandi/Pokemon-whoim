"""Stable Diffusion + ControlNet Canny + IP-Adapter pipeline.

Pikachu-ifies every Pokemon: preserves each one's shape via Canny edges,
injects Pikachu's visual appearance via IP-Adapter.
"""

import logging
import os

import torch
from PIL import Image
from tqdm import tqdm

from data.canny_utils import extract_canny_edges
from data.preprocessing import extract_alpha, rgba_to_rgb_white_bg
from inference.postprocess import restore_alpha
from utils.config import Config
from utils.device import get_device

logger = logging.getLogger(__name__)


def _resolve_dtype(use_fp16: bool, device: torch.device) -> torch.dtype:
    """Choose dtype based on device capabilities."""
    if device.type == "cuda" and use_fp16:
        return torch.float16
    return torch.float32


def load_sd_pipeline(config: Config):
    """Build the SD + ControlNet + IP-Adapter pipeline.

    Downloads models from HuggingFace on first call (~9 GB), cached thereafter.
    """
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    device = get_device()
    dtype = _resolve_dtype(config.use_fp16, device)

    logger.info("Loading ControlNet: %s", config.controlnet_model_id)
    controlnet = ControlNetModel.from_pretrained(
        config.controlnet_model_id,
        torch_dtype=dtype,
    )

    vae = None
    if config.vae_model_id:
        logger.info("Loading VAE: %s", config.vae_model_id)
        vae = AutoencoderKL.from_pretrained(
            config.vae_model_id,
            torch_dtype=dtype,
        )

    logger.info("Loading SD pipeline: %s", config.sd_model_id)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        config.sd_model_id,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    logger.info("Loading IP-Adapter: %s", config.ip_adapter_weight)
    pipeline.load_ip_adapter(
        config.ip_adapter_repo,
        subfolder="models",
        weight_name=config.ip_adapter_weight,
    )
    pipeline.set_ip_adapter_scale(config.ip_adapter_scale)

    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config,
    )

    pipeline.to(device)

    if device.type == "mps":
        # MPS unified memory — offload aggressively
        pipeline.enable_sequential_cpu_offload()

    logger.info("Pipeline ready on %s with dtype %s", device, dtype)
    return pipeline


def load_pikachu_reference(config: Config) -> Image.Image:
    """Load and prepare the Pikachu reference image for IP-Adapter."""
    img = Image.open(config.style_image).convert("RGBA")
    rgb = rgba_to_rgb_white_bg(img)
    return rgb.resize((config.image_size, config.image_size), Image.LANCZOS)


def generate_single(
    pipeline,
    target_rgb: Image.Image,
    pikachu_ref: Image.Image,
    config: Config,
    original_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Generate a single Pikachu-ified Pokemon image.

    Args:
        pipeline: The loaded SD+ControlNet+IP-Adapter pipeline.
        target_rgb: Target Pokemon as RGB PIL image.
        pikachu_ref: Pikachu reference image for IP-Adapter.
        config: Config with generation parameters.
        original_size: (W, H) of the original image for upscaling.

    Returns:
        Generated RGB PIL image.
    """
    target_resized = target_rgb.resize(
        (config.image_size, config.image_size), Image.LANCZOS,
    )

    canny_image = extract_canny_edges(
        target_resized,
        low_threshold=config.canny_low_threshold,
        high_threshold=config.canny_high_threshold,
    )

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(config.seed)

    result = pipeline(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        image=canny_image,
        ip_adapter_image=pikachu_ref,
        controlnet_conditioning_scale=config.controlnet_conditioning_scale,
        generator=generator,
    ).images[0]

    if original_size:
        result = result.resize(original_size, Image.LANCZOS)

    return result


def batch_sd_transfer(config: Config | None = None):
    """Apply Pikachu style transfer to all Pokemon images using SD pipeline."""
    if config is None:
        config = Config()

    pipeline = load_sd_pipeline(config)
    pikachu_ref = load_pikachu_reference(config)

    image_dir = config.image_dir
    output_dir = config.sd_output_dir
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png") and os.path.isfile(os.path.join(image_dir, f))
    ])

    # Resume: skip already-processed files
    done_files = set()
    if os.path.exists(output_dir):
        done_files = {
            f.replace("_pikachu_sd.png", ".png")
            for f in os.listdir(output_dir)
            if f.endswith("_pikachu_sd.png")
        }

    pending = [f for f in image_files if f not in done_files]

    print(f"\nSD Pikachu-ification: {len(pending)} pending / {len(image_files)} total")
    print(f"  Model: {config.sd_model_id}")
    print(f"  ControlNet: {config.controlnet_model_id}")
    print(f"  IP-Adapter scale: {config.ip_adapter_scale}")
    print(f"  ControlNet scale: {config.controlnet_conditioning_scale}")
    print(f"  Steps: {config.num_inference_steps}")
    print(f"  Output: {output_dir}\n")

    for filename in tqdm(pending, desc="SD transfer"):
        try:
            _process_one(pipeline, pikachu_ref, config, image_dir, output_dir, filename)
        except Exception as e:
            logger.error("Failed on %s: %s", filename, e)
            fail_path = output_dir / f"{os.path.splitext(filename)[0]}_FAILED.txt"
            fail_path.write_text(str(e))

    print(f"\nDone! Results in {output_dir}")


def _process_one(pipeline, pikachu_ref, config, image_dir, output_dir, filename):
    """Process a single Pokemon image through the SD pipeline."""
    path = os.path.join(image_dir, filename)
    img = Image.open(path).convert("RGBA")

    original_size = img.size  # (W, H)
    alpha_mask = extract_alpha(img)

    rgb = rgba_to_rgb_white_bg(img)

    result = generate_single(pipeline, rgb, pikachu_ref, config, original_size)
    result = restore_alpha(result, alpha_mask, threshold=128)

    base = os.path.splitext(filename)[0]
    result.save(output_dir / f"{base}_pikachu_sd.png", "PNG")
