"""
src/pipeline.py  –  FLUX.1-Kontext-dev Latent Interpolation

Memory strategy for 16 GB VRAM:
  - VAE mode  : only loads the VAE (~300 MB) – default, very fast.
  - Full mode : loads transformer with NF4 4-bit quant + CPU offload (~12 GB).
"""

import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"


# ───────────────────────────────────────────────
def get_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


# ───────────────────────────────────────────────
def load_vae(device: str = "cuda") -> AutoencoderKL:
    """
    Load only the FLUX.1-Kontext VAE (~300 MB).
    Runs fine on any GPU with >=2 GB VRAM.
    """
    print(f"[VAE] Loading from {MODEL_ID} ...")
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=get_dtype(device),
    ).to(device)
    vae.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"[VAE] Loaded. VRAM used: {vram:.2f} GB")
    return vae


# ───────────────────────────────────────────────
def load_full_pipeline(device: str = "cuda"):
    """
    Load full FLUX.1-Kontext-dev pipeline with NF4 4-bit quantization.
    Fits in ~14 GB VRAM (safe for 16 GB cards like T4, A10, RTX 3080Ti).
    Returns a FluxKontextPipeline.
    """
    from diffusers import FluxKontextPipeline
    from diffusers import BitsAndBytesConfig
    from diffusers.models import FluxTransformer2DModel

    print("[Full Pipeline] Applying NF4 4-bit quantization to transformer ...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )

    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    # Offload text encoders to CPU to save ~3 GB VRAM
    pipe.enable_model_cpu_offload()
    print("[Full Pipeline] Ready with CPU offload.")
    return pipe


# ───────────────────────────────────────────────
def preprocess(image_path: str, size: int = 512) -> torch.Tensor:
    """Load image, resize to size x size, normalize to [-1, 1]."""
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def encode(vae: AutoencoderKL, tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Encode an image tensor to its latent mean (no noise)."""
    tensor = tensor.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.mean
    return latent


def decode(vae: AutoencoderKL, latent: torch.Tensor) -> Image.Image:
    """Decode a latent tensor to a PIL RGB image."""
    with torch.no_grad():
        decoded = vae.decode(latent).sample
    pixel = decoded.squeeze(0).permute(1, 2, 0).cpu().float()
    pixel = ((pixel + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    return Image.fromarray(pixel)


# ───────────────────────────────────────────────
def interpolate(
    vae: AutoencoderKL,
    image_a: str,
    image_b: str,
    alpha: float = 0.5,
    size: int = 512,
    device: str = "cuda",
) -> tuple:
    """
    Latent-space interpolation between image_a and image_b.

    Args:
        vae    : loaded FLUX VAE
        image_a: path to first image  (e.g., apple)
        image_b: path to second image (e.g., banana)
        alpha  : blend factor  0=A, 0.5=midpoint, 1=B
        size   : resize both images to size x size
        device : 'cuda' or 'cpu'

    Returns:
        (pil_a, pil_b, pil_midpoint)
    """
    print(f"\nEncoding [{image_a}]...")
    la = encode(vae, preprocess(image_a, size), device)

    print(f"Encoding [{image_b}]...")
    lb = encode(vae, preprocess(image_b, size), device)

    print(f"Interpolating at alpha={alpha} ...")
    latent_mid = (1.0 - alpha) * la + alpha * lb

    print("Decoding midpoint latent ...")
    img_mid = decode(vae, latent_mid)

    pil_a = Image.open(image_a).convert("RGB").resize((size, size))
    pil_b = Image.open(image_b).convert("RGB").resize((size, size))

    return pil_a, pil_b, img_mid
