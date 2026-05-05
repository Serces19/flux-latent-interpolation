"""
src/pipeline.py  –  FLUX.2-klein-4B Latent Interpolation

Two modes:
  1. VAE-only  : encodes images → lerp latents → decode  (no text, very fast)
  2. Klein full: uses Flux2KleinPipeline with multi-reference image editing
                 to semantically blend two images at a given alpha.

Model: black-forest-labs/FLUX.2-klein-4B
VRAM : ~13 GB (fits RTX 3090 / 4070 / Colab T4 with cpu offload)
"""

import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


# ───────────────────────────────────────────────
# Mode 1 – VAE only (no transformer, fast, pure math lerp)
# ───────────────────────────────────────────────
def load_vae(device: str = "cuda") -> AutoencoderKL:
    """
    Load only the FLUX.2-klein VAE (~300 MB).
    Used for the pure latent-space interpolation (Mode 1).
    """
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[VAE] Loading from {MODEL_ID} ...")
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.eval()
    if device == "cuda":
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"[VAE] Loaded. VRAM: {vram:.2f} GB")
    return vae


def preprocess(image_path: str, size: int = 512) -> torch.Tensor:
    """Open image, resize, normalize to [-1, 1], add batch dim."""
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def encode(vae: AutoencoderKL, tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Encode image tensor to latent mean (deterministic, no noise)."""
    tensor = tensor.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        return vae.encode(tensor).latent_dist.mean


def decode(vae: AutoencoderKL, latent: torch.Tensor) -> Image.Image:
    """Decode latent tensor to PIL RGB image."""
    with torch.no_grad():
        decoded = vae.decode(latent).sample
    pixel = decoded.squeeze(0).permute(1, 2, 0).cpu().float()
    pixel = ((pixel + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    return Image.fromarray(pixel)


def interpolate_vae(
    vae: AutoencoderKL,
    image_a: str,
    image_b: str,
    alpha: float = 0.5,
    size: int = 512,
    device: str = "cuda",
) -> tuple:
    """
    Mode 1: Pure VAE latent-space linear interpolation.
    Fast, deterministic, no text conditioning.

    Returns (pil_a, pil_b, pil_midpoint)
    """
    print(f"[VAE lerp] Encoding A: {image_a}")
    la = encode(vae, preprocess(image_a, size), device)

    print(f"[VAE lerp] Encoding B: {image_b}")
    lb = encode(vae, preprocess(image_b, size), device)

    print(f"[VAE lerp] Interpolating alpha={alpha}")
    latent_mid = (1.0 - alpha) * la + alpha * lb

    print("[VAE lerp] Decoding...")
    img_mid = decode(vae, latent_mid)

    pil_a = Image.open(image_a).convert("RGB").resize((size, size))
    pil_b = Image.open(image_b).convert("RGB").resize((size, size))
    return pil_a, pil_b, img_mid


# ───────────────────────────────────────────────
# Mode 2 – Full FLUX.2-klein pipeline (semantic blending)
# ───────────────────────────────────────────────
def load_klein_pipeline(device: str = "cuda"):
    """
    Load full FLUX.2-klein-4B pipeline.
    ~13 GB VRAM with cpu_offload enabled (safe for 16 GB cards).
    Requires diffusers from git:
      pip install git+https://github.com/huggingface/diffusers.git
    """
    from diffusers import Flux2KleinPipeline

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[Klein] Loading {MODEL_ID} in bfloat16 ...")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()  # text encoders go to CPU -> saves ~3 GB VRAM
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
    Mode 2: Semantic blending via FLUX.2-klein multi-reference editing.

    Uses both images as reference conditions plus a text prompt that
    describes the desired midpoint blend.

    Returns (pil_a, pil_b, pil_midpoint)
    """
    pil_a = Image.open(image_a).convert("RGB").resize((size, size))
    pil_b = Image.open(image_b).convert("RGB").resize((size, size))

    # Build prompt based on alpha
    pct_a = int((1 - alpha) * 100)
    pct_b = int(alpha * 100)
    prompt = (
        f"A hybrid object that is {pct_a}% the first reference image "
        f"and {pct_b}% the second reference image. "
        "Seamlessly blend the shapes, colors, and textures of both objects "
        "into a single coherent image. High quality, photorealistic."
    )
    print(f"[Klein] Prompt: {prompt}")

    generator = torch.Generator().manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=[pil_a, pil_b],          # multi-reference input
        height=size,
        width=size,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )
    img_mid = result.images[0]
    return pil_a, pil_b, img_mid
