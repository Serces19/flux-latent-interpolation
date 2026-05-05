"""
Visualization utilities for the interpolation pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path
import numpy as np


def save_comparison(
    img_a: Image.Image,
    img_b: Image.Image,
    img_mid: Image.Image,
    label_a: str = "Image A",
    label_b: str = "Image B",
    label_mid: str = "Midpoint",
    output_path: str = "results/comparison.png",
) -> str:
    """Save a side-by-side comparison figure."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, img, title in zip(axes, [img_a, img_mid, img_b], [label_a, label_mid, label_b]):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=14, pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle(f"{label_a}  →  {label_mid}  →  {label_b}",
                 color="#e0e0e0", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Comparison saved to: {output_path}")
    return output_path


def save_alpha_sweep(
    vae,
    image_a: str,
    image_b: str,
    alphas: list = None,
    size: int = 512,
    device: str = "cuda",
    output_path: str = "results/alpha_sweep.png",
) -> str:
    """Generate and save a sweep of alpha values from A to B."""
    from src.pipeline import preprocess, encode, decode

    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    ta = preprocess(image_a, size)
    tb = preprocess(image_b, size)
    la = encode(vae, ta, device)
    lb = encode(vae, tb, device)

    n = len(alphas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, alpha in zip(axes, alphas):
        latent = (1 - alpha) * la + alpha * lb
        img = decode(vae, latent)
        ax.imshow(img)
        ax.set_title(f"α={alpha:.2f}", color="white", fontsize=13)
        ax.axis("off")

    plt.suptitle("Latent Space Alpha Sweep", color="#e0e0e0", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Alpha sweep saved to: {output_path}")
    return output_path
