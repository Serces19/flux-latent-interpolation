"""
run.py  –  FLUX.2-klein-4B Latent Interpolation — Main Entry Point

Image input options:
  --image_a / --image_b  : provide local file paths directly
  --upload               : prompt Colab file upload dialog (2 images)

Modes:
  --mode vae   : VAE latent lerp (fast, ~300 MB, no transformer)
  --mode klein : Full Flux2KleinPipeline semantic blend (~13 GB VRAM)

Usage examples:
  # Colab interactive upload
  python run.py --upload --mode vae --sweep
  python run.py --upload --mode klein --steps 4

  # Provide paths directly
  python run.py --image_a my_apple.jpg --image_b my_banana.jpg --mode vae
"""

import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}  ({vram:.1f} GB VRAM)")
        return "cuda"
    print("WARNING: No GPU found – using CPU (slow for 'klein' mode).")
    return "cpu"


# ─────────────────────────────────────────────────────────────────
def upload_two_images() -> tuple[str, str]:
    """
    Use google.colab.files.upload() to let the user pick two images.
    Returns (path_a, path_b) – saved under assets/.
    """
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError(
            "--upload only works inside Google Colab.\n"
            "Use --image_a and --image_b to provide local paths instead."
        )
    from IPython.display import display, Image as IPImage

    Path("assets").mkdir(exist_ok=True)

    print("=" * 55)
    print("  UPLOAD  Image A  (your first object, e.g. apple) ")
    print("=" * 55)
    up_a = files.upload()
    if not up_a:
        raise ValueError("No file uploaded for Image A.")
    fname_a = list(up_a.keys())[0]
    path_a = f"assets/image_a_{fname_a}"
    with open(path_a, "wb") as f:
        f.write(up_a[fname_a])
    print(f"✓ Image A saved: {path_a}")
    display(IPImage(path_a, width=256))

    print()
    print("=" * 55)
    print("  UPLOAD  Image B  (your second object, e.g. banana)")
    print("=" * 55)
    up_b = files.upload()
    if not up_b:
        raise ValueError("No file uploaded for Image B.")
    fname_b = list(up_b.keys())[0]
    path_b = f"assets/image_b_{fname_b}"
    with open(path_b, "wb") as f:
        f.write(up_b[fname_b])
    print(f"✓ Image B saved: {path_b}")
    display(IPImage(path_b, width=256))

    return path_a, path_b


# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2-klein-4B Latent Interpolation",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Image input
    img_group = parser.add_mutually_exclusive_group()
    img_group.add_argument("--upload", action="store_true",
                           help="Upload two images via Colab file picker")
    img_group.add_argument("--image_a", default=None,
                           help="Path to first image (use with --image_b)")
    parser.add_argument("--image_b", default=None,
                        help="Path to second image")

    # Pipeline options
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend factor: 0=A, 0.5=midpoint, 1=B  [default: 0.5]")
    parser.add_argument("--size", type=int, default=512,
                        help="Resize for VAE mode (pixels, square)  [default: 512]")
    parser.add_argument("--mode", choices=["vae", "klein"], default="vae",
                        help="'vae' = fast latent lerp\n'klein' = full FLUX.2-klein-4B  [default: vae]")
    parser.add_argument("--steps", type=int, default=4,
                        help="Diffusion steps (klein mode only)  [default: 4]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true",
                        help="Also generate alpha sweep strip [0,0.25,0.5,0.75,1] (VAE only)")
    parser.add_argument("--output_dir", default="results",
                        help="Output folder  [default: results]")
    args = parser.parse_args()

    # ── Resolve image paths
    if args.upload:
        path_a, path_b = upload_two_images()
    elif args.image_a and args.image_b:
        path_a, path_b = args.image_a, args.image_b
        if not Path(path_a).exists():
            raise FileNotFoundError(f"Image A not found: {path_a}")
        if not Path(path_b).exists():
            raise FileNotFoundError(f"Image B not found: {path_b}")
    else:
        parser.error("Provide either --upload  OR  both --image_a and --image_b")

    # ── Derive friendly labels from filenames
    label_a = Path(path_a).stem.replace("image_a_", "").split(".")[0].capitalize()
    label_b = Path(path_b).stem.replace("image_b_", "").split(".")[0].capitalize()

    device = get_device()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from src.visualize import save_comparison, save_alpha_sweep

    # ── VAE mode (fast, pure math)
    if args.mode == "vae":
        from src.pipeline import load_vae, interpolate_vae
        vae = load_vae(device=device)
        img_a, img_b, img_mid = interpolate_vae(
            vae, path_a, path_b,
            alpha=args.alpha, size=args.size, device=device,
        )
        if args.sweep:
            save_alpha_sweep(
                vae=vae,
                image_a=path_a, image_b=path_b,
                alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
                size=args.size, device=device,
                output_path=f"{args.output_dir}/alpha_sweep.png",
            )

    # ── Klein mode (full FLUX.2-klein-4B)
    elif args.mode == "klein":
        from src.pipeline import load_klein_pipeline, interpolate_klein
        pipe = load_klein_pipeline(device=device)
        img_a, img_b, img_mid = interpolate_klein(
            pipe, path_a, path_b,
            alpha=args.alpha, size=1024,
            steps=args.steps, seed=args.seed,
        )

    # ── Save outputs
    mid_path = f"{args.output_dir}/midpoint_{args.mode}_alpha{args.alpha:.2f}.png"
    img_mid.save(mid_path)
    print(f"\nMidpoint saved: {mid_path}")

    save_comparison(
        img_a, img_b, img_mid,
        label_a=label_a,
        label_b=label_b,
        label_mid=f"{label_a}-{label_b}  α={args.alpha} [{args.mode}]",
        output_path=f"{args.output_dir}/comparison_{args.mode}.png",
    )

    # ── Display inline if in Colab
    try:
        from IPython.display import display, Image as IPImage
        print("\n── Comparison ──")
        display(IPImage(f"{args.output_dir}/comparison_{args.mode}.png"))
        if args.sweep and args.mode == "vae":
            print("\n── Alpha Sweep ──")
            display(IPImage(f"{args.output_dir}/alpha_sweep.png"))
    except Exception:
        pass

    print("\n✓ Done! Check the results/ folder.")


if __name__ == "__main__":
    main()
