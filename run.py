"""
run.py  –  FLUX.2-klein-4B Latent Interpolation — Main Entry Point

Modes:
  --mode vae   : VAE latent lerp (fast, only ~300 MB, no transformer)
  --mode klein : Full Flux2KleinPipeline semantic blend (~13 GB VRAM)

Usage (Google Colab / terminal):
  python run.py --mode vae   --use_sample_images --sweep
  python run.py --mode klein --use_sample_images --steps 4
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
    print("WARNING: No GPU found, using CPU (very slow for 'klein' mode).")
    return "cpu"


def download_sample_images():
    """
    Download apple and banana sample images.
    Uses a proper User-Agent so Wikipedia doesn't block the request.
    Falls back to generating synthetic colored images if download fails.
    """
    import urllib.request
    from PIL import Image as _Image

    samples = {
        "assets/apple.jpg": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "1/15/Red_Apple.jpg/512px-Red_Apple.jpg"
        ),
        "assets/banana.jpg": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "f/f4/Banana_fruit.jpg/512px-Banana_fruit.jpg"
        ),
    }
    # Fallback synthetic colors if all URLs fail
    fallbacks = {
        "assets/apple.jpg":  (200,  40,  40),   # red
        "assets/banana.jpg": (240, 210,  30),    # yellow
    }

    Path("assets").mkdir(exist_ok=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; flux-klein-pipeline/1.0; "
            "+https://github.com/Serces19/flux-latent-interpolation)"
        )
    }

    for path, url in samples.items():
        if Path(path).exists():
            print(f"  ✓ {path} already exists")
            continue
        try:
            print(f"Downloading {path} ...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
                f.write(resp.read())
            print(f"  ✓ saved {path}")
        except Exception as e:
            print(f"  ⚠ Download failed ({e}), generating synthetic image ...")
            color = fallbacks[path]
            img = _Image.new("RGB", (512, 512), color=color)
            # Add a simple gradient so it's not a flat solid
            arr = __import__("numpy").array(img, dtype="float32")
            for i in range(512):
                arr[i] = arr[i] * (0.6 + 0.4 * i / 512)
            _Image.fromarray(arr.clip(0, 255).astype("uint8")).save(path)
            print(f"  ✓ synthetic {path} saved")


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2-klein-4B Latent Interpolation"
    )
    parser.add_argument("--image_a", default="assets/apple.jpg",
                        help="Path to first image (e.g. apple)")
    parser.add_argument("--image_b", default="assets/banana.jpg",
                        help="Path to second image (e.g. banana)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend factor: 0=A, 0.5=midpoint, 1=B")
    parser.add_argument("--size", type=int, default=512,
                        help="Resize images to size×size (VAE mode)")
    parser.add_argument("--mode", choices=["vae", "klein"], default="vae",
                        help="'vae'=fast latent lerp | 'klein'=full pipeline")
    parser.add_argument("--steps", type=int, default=4,
                        help="Diffusion steps (klein mode only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true",
                        help="Generate alpha sweep strip (VAE mode only)")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--use_sample_images", action="store_true",
                        help="Auto-download apple+banana sample images")
    args = parser.parse_args()

    if args.use_sample_images:
        download_sample_images()
        args.image_a = "assets/apple.jpg"
        args.image_b = "assets/banana.jpg"

    device = get_device()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from src.visualize import save_comparison, save_alpha_sweep

    # ── Mode 1: VAE latent interpolation (fast)
    if args.mode == "vae":
        from src.pipeline import load_vae, interpolate_vae
        vae = load_vae(device=device)
        img_a, img_b, img_mid = interpolate_vae(
            vae, args.image_a, args.image_b,
            alpha=args.alpha, size=args.size, device=device,
        )
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

    # ── Mode 2: Full Flux2KleinPipeline (semantic blend)
    elif args.mode == "klein":
        from src.pipeline import load_klein_pipeline, interpolate_klein
        pipe = load_klein_pipeline(device=device)
        img_a, img_b, img_mid = interpolate_klein(
            pipe, args.image_a, args.image_b,
            alpha=args.alpha,
            size=1024,
            steps=args.steps,
            seed=args.seed,
        )

    # ── Save outputs
    mid_path = f"{args.output_dir}/midpoint_{args.mode}_alpha{args.alpha:.2f}.png"
    img_mid.save(mid_path)
    print(f"\nMidpoint saved: {mid_path}")

    save_comparison(
        img_a, img_b, img_mid,
        label_a="🍎 Apple",
        label_b="🍌 Banana",
        label_mid=f"Apple-Banana  α={args.alpha} [{args.mode}]",
        output_path=f"{args.output_dir}/comparison_{args.mode}.png",
    )

    print("\n✓ Done! Check the results/ folder.")


if __name__ == "__main__":
    main()
