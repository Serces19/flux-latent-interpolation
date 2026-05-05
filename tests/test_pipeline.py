"""
tests/test_pipeline.py  –  Unit tests for FLUX.2-klein-4B interpolation pipeline.

All tests use mock VAEs or synthetic images — NO GPU or model download required.
Run: python tests/test_pipeline.py
"""

import sys
sys.path.insert(0, ".")

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile


# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def make_image(color: tuple, path: str, size: int = 64) -> str:
    """Create a solid-color PNG at the given path."""
    img = Image.new("RGB", (size, size), color=color)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ─────────────────────────────────────────────────────────────────
# Test 1: preprocess shape and pixel range
# ─────────────────────────────────────────────────────────────────
def test_preprocess():
    from src.pipeline import preprocess

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    make_image((200, 50, 30), tmp, size=64)

    try:
        t = preprocess(tmp, size=64)
        assert t.shape == (1, 3, 64, 64), f"Wrong shape: {t.shape}"
        assert t.min().item() >= -1.05, "Values below -1"
        assert t.max().item() <= 1.05,  "Values above +1"
        print("[PASS] test_preprocess")
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────
# Test 2: alpha interpolation math (no model needed)
# ─────────────────────────────────────────────────────────────────
def test_alpha_math():
    la = torch.zeros(1, 4, 8, 8)
    lb = torch.ones(1, 4, 8, 8)

    mid_0   = (1 - 0.0) * la + 0.0 * lb
    mid_1   = (1 - 1.0) * la + 1.0 * lb
    mid_05  = (1 - 0.5) * la + 0.5 * lb

    assert torch.allclose(mid_0,  la),                    "alpha=0 should return A"
    assert torch.allclose(mid_1,  lb),                    "alpha=1 should return B"
    assert torch.allclose(mid_05, torch.full_like(la, 0.5)), "alpha=0.5 should be 0.5"
    print("[PASS] test_alpha_math")


# ─────────────────────────────────────────────────────────────────
# Test 3: decode produces a valid PIL image (MockVAE)
# ─────────────────────────────────────────────────────────────────
def test_decode_pil():
    from src.pipeline import decode

    class _Out:
        def __init__(self, t): self.sample = t

    class MockVAE:
        dtype = torch.float32
        def decode(self, z):
            return _Out(torch.zeros(1, 3, 64, 64))

    img = decode(MockVAE(), torch.zeros(1, 4, 8, 8))
    assert isinstance(img, Image.Image), "Output must be PIL.Image"
    assert img.size == (64, 64),         f"Expected 64×64, got {img.size}"
    print("[PASS] test_decode_pil")


# ─────────────────────────────────────────────────────────────────
# Test 4: comparison figure is saved correctly
# ─────────────────────────────────────────────────────────────────
def test_save_comparison():
    from src.visualize import save_comparison

    red    = Image.new("RGB", (64, 64), (220,  50,  50))
    yellow = Image.new("RGB", (64, 64), (250, 220,  20))
    mid    = Image.new("RGB", (64, 64), (235, 135,  35))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test_comparison.png")
        save_comparison(red, yellow, mid,
                        label_a="Apple", label_b="Banana",
                        label_mid="Apple-Banana", output_path=out)
        assert Path(out).exists(),               "File not created"
        assert Path(out).stat().st_size > 5000,  "File too small, likely corrupt"
    print("[PASS] test_save_comparison")


# ─────────────────────────────────────────────────────────────────
# Test 5: interpolate_vae end-to-end with MockVAE
# ─────────────────────────────────────────────────────────────────
def test_interpolate_vae_mock():
    """Full VAE interpolation pass with a mock VAE (no GPU, no download)."""
    from src.pipeline import interpolate_vae

    class _LatentDist:
        def __init__(self, t): self.mean = t

    class _EncOut:
        def __init__(self, t): self.latent_dist = _LatentDist(t)

    class _DecOut:
        def __init__(self, t): self.sample = t

    class MockVAE:
        dtype = torch.float32
        def encode(self, x):
            # Return a latent the same spatial size as input / 8
            b, c, h, w = x.shape
            return _EncOut(torch.zeros(b, 4, h // 8, w // 8))
        def decode(self, z):
            b, c, h, w = z.shape
            return _DecOut(torch.zeros(b, 3, h * 8, w * 8))

    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = make_image((220,  50,  50), f"{tmpdir}/a.png", 64)
        path_b = make_image((250, 220,  20), f"{tmpdir}/b.png", 64)

        vae = MockVAE()
        # Monkey-patch interpolate_vae to use mock
        import src.pipeline as pl
        orig_encode = pl.encode
        orig_decode = pl.decode
        pl.encode = lambda v, t, d: v.encode(t).latent_dist.mean
        pl.decode = lambda v, l: Image.fromarray(
            ((v.decode(l).sample.squeeze(0).permute(1, 2, 0).numpy() + 1) * 127.5)
            .clip(0, 255).astype(np.uint8))

        pil_a, pil_b, pil_mid = interpolate_vae(
            vae, path_a, path_b, alpha=0.5, size=64, device="cpu"
        )

        pl.encode = orig_encode
        pl.decode = orig_decode

    assert isinstance(pil_a,   Image.Image), "pil_a must be PIL"
    assert isinstance(pil_b,   Image.Image), "pil_b must be PIL"
    assert isinstance(pil_mid, Image.Image), "pil_mid must be PIL"
    print("[PASS] test_interpolate_vae_mock")


# ─────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== FLUX.2-klein-4B Pipeline Tests ===\n")
    test_preprocess()
    test_alpha_math()
    test_decode_pil()
    test_save_comparison()
    test_interpolate_vae_mock()
    print("\n✓ All 5 tests passed!")
