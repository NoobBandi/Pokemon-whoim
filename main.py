"""Pokemon Pikachu-ifier — SD + ControlNet + IP-Adapter"""

import argparse
import os

from utils.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Pokemon Pikachu-ifier — Who's That Pokemon? It's always Pikachu!"
    )
    parser.add_argument(
        "mode",
        choices=["train", "transfer", "single", "lab_transfer", "sd_transfer", "sd_single"],
        help=(
            "sd_transfer: batch SD+ControlNet+IP-Adapter | "
            "sd_single: one image via SD pipeline | "
            "lab_transfer: batch LAB color | "
            "train/transfer/single: legacy AdaIN modes"
        ),
    )
    parser.add_argument("--input", help="Input image filename for single/sd_single mode")
    parser.add_argument("--alpha", type=float, default=1.0, help="Style strength (0.0-1.0)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (legacy)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (legacy)")
    parser.add_argument("--checkpoint", help="Path to decoder checkpoint (legacy)")

    # SD pipeline parameters
    parser.add_argument("--ip-scale", type=float, default=0.7,
                        help="IP-Adapter influence (0.0-1.0)")
    parser.add_argument("--controlnet-scale", type=float, default=0.8,
                        help="ControlNet conditioning strength")
    parser.add_argument("--steps", type=int, default=25,
                        help="Number of inference steps (20-30)")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    config = Config()
    config.alpha = args.alpha
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size

    if args.mode == "lab_transfer":
        from inference.color_transfer import batch_lab_transfer
        batch_lab_transfer(config)

    elif args.mode == "train":
        from train.trainer import train
        train(config)

    elif args.mode == "transfer":
        from inference.transfer import batch_transfer
        batch_transfer(config, checkpoint_path=args.checkpoint, alpha=args.alpha)

    elif args.mode == "single":
        if not args.input:
            parser.error("single mode requires --input")

        from torchvision import transforms
        from data.dataset import PokemonImageDataset
        from data.preprocessing import build_transform
        from data.style_utils import load_style_image
        from inference.transfer import load_model, transfer_single
        from inference.postprocess import restore_alpha
        from utils.device import get_device

        device = get_device()

        ckpt_path = args.checkpoint
        if not ckpt_path:
            ckpts = sorted(config.checkpoint_dir.glob("decoder_epoch_*.pth"))
            if not ckpts:
                parser.error("No checkpoints found. Run training first.")
            ckpt_path = str(ckpts[-1])

        model = load_model(ckpt_path, device)
        transform = build_transform(config.image_size)
        dataset = PokemonImageDataset(str(config.image_dir), transform=transform)

        idx = None
        for i, (_, fname) in enumerate(dataset):
            if fname == args.input or fname == args.input + ".png":
                idx = i
                break

        if idx is None:
            parser.error(f"Image '{args.input}' not found in dataset")

        content_tensor, filename = dataset[idx]
        style_tensor = load_style_image(
            str(config.style_image),
            target_size=config.image_size,
            imagenet_mean=config.imagenet_mean,
            imagenet_std=config.imagenet_std,
        ).to(device)

        result = transfer_single(model, content_tensor, style_tensor, device, args.alpha)
        result_pil = transforms.functional.to_pil_image(result)
        alpha_mask = dataset.get_alpha(filename)
        if alpha_mask is not None:
            result_pil = restore_alpha(result_pil, alpha_mask)

        base = os.path.splitext(filename)[0]
        out_path = config.output_dir / f"{base}_pikachu.png"
        os.makedirs(config.output_dir, exist_ok=True)
        result_pil.save(out_path, "PNG")
        print(f"Saved: {out_path}")

    elif args.mode == "sd_transfer":
        config.ip_adapter_scale = args.ip_scale
        config.controlnet_conditioning_scale = args.controlnet_scale
        config.num_inference_steps = args.steps
        config.guidance_scale = args.guidance
        config.seed = args.seed

        from inference.sd_pipeline import batch_sd_transfer
        batch_sd_transfer(config)

    elif args.mode == "sd_single":
        if not args.input:
            parser.error("sd_single mode requires --input")

        config.ip_adapter_scale = args.ip_scale
        config.controlnet_conditioning_scale = args.controlnet_scale
        config.num_inference_steps = args.steps
        config.guidance_scale = args.guidance
        config.seed = args.seed

        from PIL import Image as PILImage
        from inference.sd_pipeline import load_sd_pipeline, load_pikachu_reference, generate_single
        from data.preprocessing import extract_alpha, rgba_to_rgb_white_bg
        from inference.postprocess import restore_alpha

        # Find image in dataset
        image_path = config.image_dir / args.input
        if not image_path.exists():
            # Try with .png extension
            image_path = config.image_dir / (args.input + ".png")
        if not image_path.exists():
            parser.error(f"Image '{args.input}' not found in {config.image_dir}")

        pipeline = load_sd_pipeline(config)
        pikachu_ref = load_pikachu_reference(config)

        img = PILImage.open(image_path).convert("RGBA")
        original_size = img.size
        alpha_mask = extract_alpha(img)
        rgb = rgba_to_rgb_white_bg(img)

        result = generate_single(pipeline, rgb, pikachu_ref, config, original_size)
        result = restore_alpha(result, alpha_mask, threshold=128)

        os.makedirs(config.sd_output_dir, exist_ok=True)
        base = os.path.splitext(args.input)[0]
        out_path = config.sd_output_dir / f"{base}_pikachu_sd.png"
        result.save(out_path, "PNG")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
