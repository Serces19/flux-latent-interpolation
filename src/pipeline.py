"""
FLUX Latent Interpolation Pipeline

Loads the FLUX VAE, encodes two images into latent space,
interpolates at a given alpha, and decodes the result.
"""

import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np


MODEL_ID = "black-forest-labs/FLUX.1-schnell"
VAE_SUBFOLDER = "vae"


def load_vae(device: str = "cuda") -> AutoencoderKL:
    """Load FLUX VAE from HuggingFace."""
    print(f"Loading VAE from {MODEL_ID} ...")
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        subfolder=VAE_SUBFOLDER,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    vae.eval()
    print("VAE loaded.")
    return vae


def preprocess(image_path: str, size: int = 512) -> torch.Tensor:
    """Load image, resize, normalize to [-1, 1], add batch dim."""
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def encode(vae: AutoencoderKL, tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Encode image tensor to latent mean."""
    tensor = tensor.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        dist = vae.encode(tensor)
        # Use mean (no sampling noise)
        latent = dist.latent_dist.mean
    return latent


def decode(vae: AutoencoderKL, latent: torch.Tensor) -> Image.Image:
    """Decode latent tensor back to PIL image."""
    with torch.no_grad():
        decoded = vae.decode(latent).sample
    # Convert to uint8 image
    decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().float()
    decoded = ((decoded + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    return Image.fromarray(decoded)


def interpolate(
    vae: AutoencoderKL,
    image_a: str,
    image_b: str,
    alpha: float = 0.5,
    size: int = 512,
    device: str = "cuda",
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """
    Main interpolation function.

    Args:
        vae: loaded VAE model
        image_a: path to first image
        image_b: path to second image
        alpha: interpolation factor (0=image_a, 1=image_b, 0.5=midpoint)
        size: resize images to this square size
        device: 'cuda' or 'cpu'

    Returns:
        (img_a_pil, img_b_pil, interpolated_pil)
    """
    print(f"Encoding image A: {image_a}")
    ta = preprocess(image_a, size)
    la = encode(vae, ta, device)

    print(f"Encoding image B: {image_b}")
    tb = preprocess(image_b, size)
    lb = encode(vae, tb, device)

    print(f"Interpolating at alpha={alpha}")
    latent_mid = (1 - alpha) * la + alpha * lb

    print("Decoding interpolated latent...")
    img_out = decode(vae, latent_mid)

    img_a_pil = Image.open(image_a).convert("RGB").resize((size, size))
    img_b_pil = Image.open(image_b).convert("RGB").resize((size, size))

    return img_a_pil, img_b_pil, img_out
