"""
colab_demo.py  –  FLUX.2-klein-4B Interactive Demo for Google Colab

Paste this entire file content into a Colab cell, or run:
    %run colab_demo.py

Workflow:
  1. Upload image A (e.g. apple)
  2. Upload image B (e.g. banana)
  3. Choose mode: 'vae' (fast) or 'klein' (full FLUX.2-klein-4B)
  4. See the midpoint + comparison figure inline
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from IPython.display import display, Image as IPImage
from PIL import Image
import ipywidgets as widgets


# ─────────────────────────────────────────────────────────────────
# Step 1 – Upload images
# ─────────────────────────────────────────────────────────────────
def upload_images() -> tuple[str, str]:
    """
    Show Colab file upload widgets for image A and B.
    Returns (path_a, path_b) where the files are saved.
    """
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError("This function only works inside Google Colab.")

    Path("assets").mkdir(exist_ok=True)

    print("=" * 50)
    print("STEP 1 of 2 — Upload IMAGE A (first object, e.g. apple)")
    print("=" * 50)
    up_a = files.upload()
    if not up_a:
        raise ValueError("No file uploaded for image A.")
    name_a = list(up_a.keys())[0]
    path_a = f"assets/image_a_{name_a}"
    with open(path_a, "wb") as f:
        f.write(up_a[name_a])
    print(f"✓ Image A saved: {path_a}")
    display(IPImage(path_a, width=256))

    print()
    print("=" * 50)
    print("STEP 2 of 2 — Upload IMAGE B (second object, e.g. banana)")
    print("=" * 50)
    up_b = files.upload()
    if not up_b:
        raise ValueError("No file uploaded for image B.")
    name_b = list(up_b.keys())[0]
    path_b = f"assets/image_b_{name_b}"
    with open(path_b, "wb") as f:
        f.write(up_b[name_b])
    print(f"✓ Image B saved: {path_b}")
    display(IPImage(path_b, width=256))

    return path_a, path_b


# ─────────────────────────────────────────────────────────────────
# Step 2 – Run interpolation
# ─────────────────────────────────────────────────────────────────
def run_interpolation(
    path_a: str,
    path_b: str,
    alpha: float = 0.5,
    mode: str = "vae",
    size: int = 512,
    steps: int = 4,
    seed: int = 42,
    sweep: bool = False,
    output_dir: str = "results",
):
    """
    Run the interpolation pipeline and display results inline.

    Args:
        path_a     : path to first uploaded image
        path_b     : path to second uploaded image
        alpha      : blend factor (0=A, 0.5=midpoint, 1=B)
        mode       : 'vae' (fast) or 'klein' (full FLUX.2-klein-4B)
        size       : image resize for VAE mode
        steps      : diffusion steps for klein mode
        seed       : random seed for klein mode
        sweep      : generate alpha sweep strip (VAE mode only)
        output_dir : folder to save result images
    """
    import torch
    from src.visualize import save_comparison, save_alpha_sweep

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_str = ""
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_str = f"  ({torch.cuda.get_device_name(0)}, {vram:.1f} GB)"
    print(f"\nDevice: {device}{vram_str}")

    label_a = Path(path_a).stem.replace("image_a_", "").split(".")[0].capitalize()
    label_b = Path(path_b).stem.replace("image_b_", "").split(".")[0].capitalize()

    # ── VAE mode
    if mode == "vae":
        print(f"\n[VAE mode] Encoding and interpolating at α={alpha} ...")
        from src.pipeline import load_vae, interpolate_vae
        vae = load_vae(device=device)
        img_a, img_b, img_mid = interpolate_vae(
            vae, path_a, path_b,
            alpha=alpha, size=size, device=device,
        )
        if sweep:
            sweep_path = f"{output_dir}/alpha_sweep.png"
            save_alpha_sweep(
                vae=vae, image_a=path_a, image_b=path_b,
                alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
                size=size, device=device, output_path=sweep_path,
            )
            print("\n── Alpha Sweep ──")
            display(IPImage(sweep_path))

    # ── Klein mode
    elif mode == "klein":
        print(f"\n[Klein mode] Loading FLUX.2-klein-4B pipeline ...")
        from src.pipeline import load_klein_pipeline, interpolate_klein
        pipe = load_klein_pipeline(device=device)
        img_a, img_b, img_mid = interpolate_klein(
            pipe, path_a, path_b,
            alpha=alpha, size=1024,
            steps=steps, seed=seed,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'vae' or 'klein'.")

    # Save
    mid_path = f"{output_dir}/midpoint_{mode}_alpha{alpha:.2f}.png"
    img_mid.save(mid_path)

    cmp_path = f"{output_dir}/comparison_{mode}.png"
    save_comparison(
        img_a, img_b, img_mid,
        label_a=label_a,
        label_b=label_b,
        label_mid=f"Midpoint  α={alpha} [{mode}]",
        output_path=cmp_path,
    )

    # Display inline
    print(f"\n── Midpoint (α={alpha}) ──")
    display(IPImage(mid_path, width=400))

    print("\n── Comparison (A | Midpoint | B) ──")
    display(IPImage(cmp_path))

    print(f"\n✓ Results saved in '{output_dir}/'")
    return img_a, img_b, img_mid


# ─────────────────────────────────────────────────────────────────
# Main: run as script
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FLUX.2-klein-4B Colab Demo with file upload"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mode", choices=["vae", "klein"], default="vae")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    path_a, path_b = upload_images()
    run_interpolation(
        path_a=path_a,
        path_b=path_b,
        alpha=args.alpha,
        mode=args.mode,
        size=args.size,
        steps=args.steps,
        seed=args.seed,
        sweep=args.sweep,
        output_dir=args.output_dir,
    )
