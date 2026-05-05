"""
src/pipeline.py  –  FLUX.2-klein-4B Latent Interpolation

Model : black-forest-labs/FLUX.2-klein-4B
VRAM  : ~13 GB with enable_model_cpu_offload (safe on 16 GB)

Two modes:
  - Mode 1 VAE  : load only the VAE (~300 MB) → encode → lerp latents → decode
  - Mode 2 Klein: full Flux2KleinPipeline with multi-reference image editing
"""

import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def preprocess(image_path: str, size: int = 512) -> torch.Tensor:
    """Open image, resize, normalize to [-1, 1], add batch dim."""
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def encode(vae: AutoencoderKL, tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Encode an image tensor to its latent mean (no sampling noise)."""
    tensor = tensor.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        return vae.encode(tensor).latent_dist.mean


def decode(vae: AutoencoderKL, latent: torch.Tensor) -> Image.Image:
    """Decode a latent tensor to a PIL RGB image."""
    with torch.no_grad():
        decoded = vae.decode(latent).sample
    pixel = decoded.squeeze(0).permute(1, 2, 0).cpu().float()
    pixel = ((pixel + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    return Image.fromarray(pixel)


# ───────────────────────────────────────────────────────────────
# Mode 1: VAE only – fast, deterministic
# ───────────────────────────────────────────────────────────────
def load_vae(device: str = "cuda") -> AutoencoderKL:
    """
    Load only the FLUX.2-klein VAE (~300 MB).
    Runs on any GPU with >=2 GB VRAM.
    """
    print(f"[VAE] Loading from {MODEL_ID} ...")
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=_dtype(device)
    ).to(device)
    vae.eval()
    if device == "cuda":
        print(f"[VAE] VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return vae


def interpolate_vae(
    vae: AutoencoderKL,
    image_a: str,
    image_b: str,
    alpha: float = 0.5,
    size: int = 512,
    device: str = "cuda",
) -> tuple:
    """
    Latent-space linear interpolation via the VAE.

    Args:
        vae     : loaded AutoencoderKL
        image_a : path to first image  (e.g. apple)
        image_b : path to second image (e.g. banana)
        alpha   : 0.0 = pure A, 0.5 = midpoint, 1.0 = pure B
        size    : resize images to size × size before encoding
        device  : 'cuda' or 'cpu'

    Returns:
        (pil_a, pil_b, pil_midpoint)
    """
    print(f"[VAE] Encoding A: {image_a}")
    la = encode(vae, preprocess(image_a, size), device)

    print(f"[VAE] Encoding B: {image_b}")
    lb = encode(vae, preprocess(image_b, size), device)

    print(f"[VAE] Interpolating at alpha={alpha}")
    latent_mid = (1.0 - alpha) * la + alpha * lb

    print("[VAE] Decoding midpoint latent ...")
    img_mid = decode(vae, latent_mid)

    pil_a = Image.open(image_a).convert("RGB").resize((size, size))
    pil_b = Image.open(image_b).convert("RGB").resize((size, size))
    return pil_a, pil_b, img_mid


# ───────────────────────────────────────────────────────────────
# Mode 2: Full FLUX.2-klein-4B pipeline – semantic blending
# ───────────────────────────────────────────────────────────────
def load_klein_pipeline(device: str = "cuda"):
    """
    Load the full Flux2KleinPipeline from FLUX.2-klein-4B.
    Uses enable_model_cpu_offload() to keep VRAM under 14 GB.

    Requires diffusers from git:
        pip install git+https://github.com/huggingface/diffusers.git
    """
    from diffusers import Flux2KleinPipeline

    dtype = _dtype(device)
    print(f"[Klein] Loading {MODEL_ID} in bfloat16 ...")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    # Offloads text encoders to CPU → saves ~3 GB VRAM
    pipe.enable_model_cpu_offload()
    print("[Klein] Pipeline ready.")
    return pipe


def interpolate_klein(
    pipe,
    image_a: str,
    image_b: str,
    alpha: float = 0.5,
    size: int = 1024,
    steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Semantic blending via FLUX.2-klein multi-reference image editing.

    Passes both images as reference conditions with a text prompt
    describing the desired blend at the given alpha.

    Returns (pil_a, pil_b, pil_midpoint)
    """
    pil_a = Image.open(image_a).convert("RGB").resize((size, size))
    pil_b = Image.open(image_b).convert("RGB").resize((size, size))

    pct_a = int((1 - alpha) * 100)
    pct_b = int(alpha * 100)
    prompt = (
        f"A hybrid object that is {pct_a}% the first reference image "
        f"and {pct_b}% the second reference image. "
        "Seamlessly blend the shapes, colors, and textures of both objects "
        "into a single coherent image. Photorealistic, high quality."
    )
    print(f"[Klein] Prompt: {prompt}")

    generator = torch.Generator().manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=[pil_a, pil_b],       # multi-reference input
        height=size,
        width=size,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )
    return pil_a, pil_b, result.images[0]
