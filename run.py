"""
run.py  –  Main entry point for FLUX Latent Interpolation.

Usage (Google Colab / terminal):
    python run.py --image_a assets/apple.jpg --image_b assets/banana.jpg
    python run.py --image_a assets/apple.jpg --image_b assets/banana.jpg --sweep
"""

import argparse
import os
from pathlib import Path


def get_device():
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("WARNING: No GPU detected, using CPU (slow).")
    return "cpu"


def download_sample_images():
    """Download apple and banana sample images from public URLs."""
    import urllib.request

    samples = {
        "assets/apple.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/512px-Red_Apple.jpg",
        "assets/banana.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Chocolate-Chip-Cookies-Recipe.jpg/512px-Banana-Chocolate-Chip-Cookies-Recipe.jpg",
    }

    Path("assets").mkdir(exist_ok=True)
    for path, url in samples.items():
        if not Path(path).exists():
            print(f"Downloading {path} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  -> saved to {path}")
        else:
            print(f"  -> {path} already exists")


def main():
    parser = argparse.ArgumentParser(description="FLUX Latent Interpolation")
    parser.add_argument("--image_a", type=str, default="assets/apple.jpg")
    parser.add_argument("--image_b", type=str, default="assets/banana.jpg")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Interpolation factor: 0=A, 1=B, 0.5=midpoint")
    parser.add_argument("--size", type=int, default=512,
                        help="Image resize dimension (square)")
    parser.add_argument("--sweep", action="store_true",
                        help="Also generate alpha sweep visualization")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--use_sample_images", action="store_true",
                        help="Download and use Wikipedia apple/banana images")
    args = parser.parse_args()

    # Optionally download samples
    if args.use_sample_images or not (Path(args.image_a).exists() and Path(args.image_b).exists()):
        download_sample_images()
        args.image_a = "assets/apple.jpg"
        args.image_b = "assets/banana.jpg"

    device = get_device()

    from src.pipeline import load_vae, interpolate
    from src.visualize import save_comparison, save_alpha_sweep

    # Load VAE
    vae = load_vae(device=device)

    # Midpoint interpolation
    img_a, img_b, img_mid = interpolate(
        vae=vae,
        image_a=args.image_a,
        image_b=args.image_b,
        alpha=args.alpha,
        size=args.size,
        device=device,
    )

    # Save individual outputs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    mid_path = f"{args.output_dir}/midpoint_alpha{args.alpha:.2f}.png"
    img_mid.save(mid_path)
    print(f"Midpoint image saved: {mid_path}")

    # Save comparison
    save_comparison(
        img_a, img_b, img_mid,
        label_a="Apple",
        label_b="Banana",
        label_mid=f"Apple-Banana (\u03b1={args.alpha})",
        output_path=f"{args.output_dir}/comparison.png",
    )

    # Alpha sweep
    if args.sweep:
        save_alpha_sweep(
            vae=vae,
            image_a=args.image_a,
            image_b=args.image_b,
            alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
            size=args.size,
            device=device,
            output_path=f"{args.output_dir}/alpha_sweep.png",
        )

    print("\nDone! Check the 'results/' folder.")


if __name__ == "__main__":
    main()
