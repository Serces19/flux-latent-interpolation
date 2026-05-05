"""
run.py  –  FLUX.2-klein-4B Latent Interpolation Entry Point

Modes:
  --mode vae    : VAE latent lerp (fast, ~300 MB, no transformer needed)
  --mode klein  : Full Flux2KleinPipeline semantic blend (~13 GB VRAM)

Usage (Google Colab):
  python run.py --mode vae   --use_sample_images --sweep
  python run.py --mode klein --use_sample_images
"""

import argparse
import urllib.request
from pathlib import Path


def get_device():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}  ({vram:.1f} GB VRAM)")
        return "cuda"
    print("WARNING: No GPU, falling back to CPU (slow for 'klein' mode).")
    return "cpu"


def download_sample_images():
    samples = {
        "assets/apple.jpg": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "1/15/Red_Apple.jpg/512px-Red_Apple.jpg"
        ),
        "assets/banana.jpg": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "8/8a/Banana-Chocolate-Chip-Cookies-Recipe.jpg/"
            "512px-Banana-Chocolate-Chip-Cookies-Recipe.jpg"
        ),
    }
    Path("assets").mkdir(exist_ok=True)
    for path, url in samples.items():
        if not Path(path).exists():
            print(f"Downloading {path} ...")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"  {path} already exists.")


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein-4B Interpolation")
    parser.add_argument("--image_a", type=str, default="assets/apple.jpg")
    parser.add_argument("--image_b", type=str, default="assets/banana.jpg")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="0=image_a, 1=image_b, 0.5=midpoint")
    parser.add_argument("--size", type=int, default=512,
                        help="Image size (VAE mode). Klein mode uses 1024.")
    parser.add_argument("--mode", choices=["vae", "klein"], default="vae",
                        help="'vae' = fast latent lerp | 'klein' = full pipeline")
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps (klein mode only)")
    parser.add_argument("--sweep", action="store_true",
                        help="Generate alpha sweep (VAE mode only)")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--use_sample_images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.use_sample_images:
        download_sample_images()
        args.image_a = "assets/apple.jpg"
        args.image_b = "assets/banana.jpg"

    device = get_device()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from src.visualize import save_comparison, save_alpha_sweep

    # ── VAE mode (fast, pure math)
    if args.mode == "vae":
        from src.pipeline import load_vae, interpolate_vae
        vae = load_vae(device=device)
        img_a, img_b, img_mid = interpolate_vae(
            vae, args.image_a, args.image_b,
            alpha=args.alpha, size=args.size, device=device,
        )
        if args.sweep:
            save_alpha_sweep(
                vae=vae, image_a=args.image_a, image_b=args.image_b,
                alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
                size=args.size, device=device,
                output_path=f"{args.output_dir}/alpha_sweep.png",
            )

    # ── Klein mode (full pipeline, semantic blend)
    elif args.mode == "klein":
        from src.pipeline import load_klein_pipeline, interpolate_klein
        pipe = load_klein_pipeline(device=device)
        img_a, img_b, img_mid = interpolate_klein(
            pipe, args.image_a, args.image_b,
            alpha=args.alpha, size=1024,
            steps=args.steps, seed=args.seed,
        )

    # Save outputs
    mid_path = f"{args.output_dir}/midpoint_{args.mode}_alpha{args.alpha:.2f}.png"
    img_mid.save(mid_path)
    print(f"Midpoint saved: {mid_path}")

    save_comparison(
        img_a, img_b, img_mid,
        label_a="Apple",
        label_b="Banana",
        label_mid=f"Apple-Banana \u03b1={args.alpha} [{args.mode}]",
        output_path=f"{args.output_dir}/comparison_{args.mode}.png",
    )

    print("\nDone! Check results/ folder.")


if __name__ == "__main__":
    main()
