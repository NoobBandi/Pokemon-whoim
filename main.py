"""AdaIN Pokemon Style Transfer — 'Who's That Pokemon? It's always Pikachu!'"""

import argparse

from utils.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="AdaIN Pokemon Style Transfer — Pikachu-ify everything!"
    )
    parser.add_argument(
        "mode",
        choices=["train", "transfer", "single"],
        help="train: train the decoder | transfer: batch apply Pikachu style | single: one image",
    )
    parser.add_argument("--input", help="Input image filename for single mode")
    parser.add_argument("--alpha", type=float, default=1.0, help="Style strength (0.0-1.0)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--checkpoint", help="Path to decoder checkpoint")
    args = parser.parse_args()

    config = Config()
    config.alpha = args.alpha
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size

    if args.mode == "train":
        from train.trainer import train
        train(config)

    elif args.mode == "transfer":
        from inference.transfer import batch_transfer
        batch_transfer(config, checkpoint_path=args.checkpoint, alpha=args.alpha)

    elif args.mode == "single":
        if not args.input:
            parser.error("single mode requires --input")

        import os
        from torchvision import transforms
        from data.dataset import PokemonImageDataset
        from data.preprocessing import build_transform
        from data.style_utils import load_style_image
        from inference.transfer import load_model, transfer_single
        from inference.postprocess import restore_alpha
        from utils.device import get_device

        device = get_device()

        # Find checkpoint
        ckpt_path = args.checkpoint
        if not ckpt_path:
            ckpts = sorted(config.checkpoint_dir.glob("decoder_epoch_*.pth"))
            if not ckpts:
                parser.error("No checkpoints found. Run training first.")
            ckpt_path = str(ckpts[-1])

        model = load_model(ckpt_path, device)

        # Load the specific image
        transform = build_transform(config.image_size)
        dataset = PokemonImageDataset(str(config.image_dir), transform=transform)

        # Find the image in dataset
        idx = None
        for i, (_, fname) in enumerate(dataset):
            if fname == args.input:
                idx = i
                break
        if idx is None:
            # Try with .png extension
            for i, (_, fname) in enumerate(dataset):
                if fname == args.input + ".png" or fname == args.input:
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


if __name__ == "__main__":
    main()
