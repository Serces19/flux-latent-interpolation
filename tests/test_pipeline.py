"""
tests/test_pipeline.py

Unit and integration tests for the FLUX Latent Interpolation pipeline.
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os


# ───────────────────────────────────────────────
# Helper: create a tiny synthetic image
# ───────────────────────────────────────────────
def make_test_image(color: tuple, path: str, size: int = 64) -> str:
    img = Image.new("RGB", (size, size), color=color)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ───────────────────────────────────────────────
# Test 1: preprocess
# ───────────────────────────────────────────────
def test_preprocess_shape_and_range():
    from src.pipeline import preprocess

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    make_test_image((200, 50, 30), tmp, size=64)

    try:
        t = preprocess(tmp, size=64)
        assert t.shape == (1, 3, 64, 64), f"Bad shape: {t.shape}"
        assert t.min().item() >= -1.1, "Min below -1"
        assert t.max().item() <= 1.1, "Max above 1"
        print("[PASS] test_preprocess_shape_and_range")
    finally:
        os.unlink(tmp)


# ───────────────────────────────────────────────
# Test 2: alpha interpolation math
# ───────────────────────────────────────────────
def test_alpha_interpolation_math():
    la = torch.zeros(1, 4, 8, 8)
    lb = torch.ones(1, 4, 8, 8)

    # alpha=0 -> all zeros
    mid = (1 - 0.0) * la + 0.0 * lb
    assert torch.allclose(mid, la), "alpha=0 should return la"

    # alpha=1 -> all ones
    mid = (1 - 1.0) * la + 1.0 * lb
    assert torch.allclose(mid, lb), "alpha=1 should return lb"

    # alpha=0.5 -> all 0.5
    mid = (1 - 0.5) * la + 0.5 * lb
    expected = torch.full_like(la, 0.5)
    assert torch.allclose(mid, expected), "alpha=0.5 should return 0.5"

    print("[PASS] test_alpha_interpolation_math")


# ───────────────────────────────────────────────
# Test 3: decode produces valid PIL image
# ───────────────────────────────────────────────
def test_decode_output_is_valid_image():
    """
    Use a MockVAE so we don't need actual model weights.
    """
    from src.pipeline import decode

    class MockDecodeOutput:
        def __init__(self, t):
            self.sample = t

    class MockVAE:
        dtype = torch.float32

        def decode(self, z):
            # Return a fake decoded tensor in [-1, 1]
            fake = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
            return MockDecodeOutput(fake)

    vae = MockVAE()
    fake_latent = torch.zeros(1, 4, 8, 8)
    img = decode(vae, fake_latent)

    assert isinstance(img, Image.Image), "Output should be PIL Image"
    assert img.size == (64, 64), f"Expected 64x64, got {img.size}"
    print("[PASS] test_decode_output_is_valid_image")


# ───────────────────────────────────────────────
# Test 4: comparison figure saved correctly
# ───────────────────────────────────────────────
def test_save_comparison_creates_file():
    from src.visualize import save_comparison

    red = Image.new("RGB", (64, 64), (220, 50, 50))
    yellow = Image.new("RGB", (64, 64), (250, 220, 20))
    mid = Image.new("RGB", (64, 64), (235, 135, 35))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "comparison.png")
        save_comparison(red, yellow, mid,
                        label_a="Apple", label_b="Banana",
                        label_mid="Apple-Banana",
                        output_path=out)
        assert Path(out).exists(), "Comparison file not created"
        assert Path(out).stat().st_size > 1000, "File looks too small"
        print("[PASS] test_save_comparison_creates_file")


# ───────────────────────────────────────────────
# Run all tests
# ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Running FLUX Interpolation Tests ===")
    test_preprocess_shape_and_range()
    test_alpha_interpolation_math()
    test_decode_output_is_valid_image()
    test_save_comparison_creates_file()
    print("\nAll tests passed!")
