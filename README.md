# FLUX Latent Interpolation Pipeline

> Find the **midpoint in latent space** between two images using the FLUX VAE.  
> Example: 🍎 Apple + 🍌 Banana = 🍌🍎 Apple-Banana hybrid.

---

## Quick Start (Google Colab)

Open a new Colab notebook, enable a **T4 GPU** runtime, and paste:

```python
# ── 1. Clone repo
!git clone https://github.com/Serces19/flux-latent-interpolation.git
%cd flux-latent-interpolation

# ── 2. Install dependencies
!pip install -q -r requirements.txt

# ── 3. Run tests (no GPU needed)
!python tests/test_pipeline.py

# ── 4. Run full pipeline (downloads apple & banana, runs interpolation)
!python run.py --use_sample_images --sweep

# ── 5. Display results inside Colab
from IPython.display import Image as IPImage
IPImage('results/comparison.png')
```

---

## What it Does

```
 Apple Image  ─► VAE Encode ─► Latent A ──┐
                                            ├─► lerp(α=0.5) ─► Latent Mid ─► VAE Decode ─► 🍌🍎
 Banana Image ─► VAE Encode ─► Latent B ──┘
```

1. **Load** FLUX.1-schnell VAE (encoder + decoder)
2. **Encode** both images into 4-channel latent maps
3. **Interpolate**: `latent_mid = (1-α) * latent_A + α * latent_B`
4. **Decode** the midpoint latent back to a pixel image
5. **Save** comparison and optional alpha sweep

---

## CLI Options

```bash
python run.py \
  --image_a assets/apple.jpg \
  --image_b assets/banana.jpg \
  --alpha 0.5 \
  --size 512 \
  --sweep \
  --output_dir results
```

| Flag | Default | Description |
|---|---|---|
| `--image_a` | `assets/apple.jpg` | Path to first image |
| `--image_b` | `assets/banana.jpg` | Path to second image |
| `--alpha` | `0.5` | Blend factor (0=A, 1=B) |
| `--size` | `512` | Resize to NxN before encoding |
| `--sweep` | off | Also generate 5-step alpha sweep |
| `--use_sample_images` | off | Auto-download Wikipedia images |

---

## Project Structure

```
flux-latent-interpolation/
├── src/
│   ├── pipeline.py      # VAE load, encode, decode, interpolate
│   └── visualize.py     # Comparison & alpha sweep figures
├── tests/
│   └── test_pipeline.py # Unit tests (no GPU required)
├── run.py               # Main entry point
├── requirements.txt
└── README.md
```

---

## Outputs

| File | Description |
|---|---|
| `results/comparison.png` | Side-by-side: Apple \| Midpoint \| Banana |
| `results/midpoint_alpha0.50.png` | Standalone midpoint image |
| `results/alpha_sweep.png` | α ∈ {0, 0.25, 0.5, 0.75, 1.0} strip |

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (T4 on Colab is free)
- HuggingFace access to `black-forest-labs/FLUX.1-schnell` (free, no token needed)

---

## Running Tests

Tests use mock VAEs — **no GPU or model download required**:

```bash
python tests/test_pipeline.py
```

Expected output:
```
=== Running FLUX Interpolation Tests ===
[PASS] test_preprocess_shape_and_range
[PASS] test_alpha_interpolation_math
[PASS] test_decode_output_is_valid_image
[PASS] test_save_comparison_creates_file

All tests passed!
```
