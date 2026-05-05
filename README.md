# FLUX.2-klein-4B Latent Interpolation Pipeline

> Encuentra el **punto medio en espacio latente** entre dos imágenes.  
> Ejemplo: 🍎 Manzana + 🍌 Banana = 🍌🍎 Apple-Banana híbrido.

**Modelo**: [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)  
**Licencia**: Apache 2.0 | **VRAM**: ~13 GB (seguro en 16 GB)

---

## ⚡ Quick Start — Google Colab (GPU T4/A10)

Abre un nuevo Colab notebook, activa una **GPU T4** y ejecuta:

```python
# 1. Instalar diffusers desde git (necesario para Flux2KleinPipeline)
!pip install -q git+https://github.com/huggingface/diffusers.git
!pip install -q torch torchvision accelerate transformers Pillow matplotlib

# 2. Clonar el repo
!git clone https://github.com/Serces19/flux-latent-interpolation.git
%cd flux-latent-interpolation

# 3. Tests (sin GPU, sin descargar el modelo)
!python tests/test_pipeline.py

# 4a. Modo VAE – solo carga VAE (~300 MB), muy rápido
!python run.py --mode vae --use_sample_images --sweep

# 4b. Modo Klein – pipeline completo FLUX.2-klein-4B (~13 GB VRAM)
!python run.py --mode klein --use_sample_images --steps 4

# 5. Visualizar resultados
from IPython.display import Image as IPImage, display
display(IPImage('results/comparison_vae.png'))
display(IPImage('results/comparison_klein.png'))
```

---

## Arquitectura del pipeline

```
Modo VAE (rápido, determinístico):
  🍎 Apple  → VAE Encode → Latent A ──┐
                                       ├─ lerp(α=0.5) → Decode → 🍌🍎
  🍌 Banana → VAE Encode → Latent B ──┘

Modo Klein (semántico, FLUX transformer 4B):
  🍎 Apple  ──┐
               ├─ Flux2KleinPipeline (multi-reference + prompt) → 🍌🍎
  🍌 Banana ──┘
```

---

## Modos de operación

| Modo | VRAM | Tiempo | Descripción |
|---|---|---|---|
| `--mode vae` | ~0.5 GB | <5 s | Interpolación matemática en latentes |
| `--mode klein` | ~13 GB | ~10-30 s | Blending semántico con transformer 4B |

---

## CLI

```bash
python run.py \
  --image_a assets/apple.jpg \
  --image_b assets/banana.jpg \
  --alpha 0.5 \
  --mode klein \
  --steps 4 \
  --seed 42 \
  --output_dir results
```

| Flag | Default | Descripción |
|---|---|---|
| `--image_a` | `assets/apple.jpg` | Primera imagen |
| `--image_b` | `assets/banana.jpg` | Segunda imagen |
| `--alpha` | `0.5` | Factor de mezcla (0=A, 1=B) |
| `--mode` | `vae` | `vae` o `klein` |
| `--steps` | `4` | Pasos de inferencia (solo klein) |
| `--sweep` | off | Genera strip α∈{0,0.25,0.5,0.75,1} (solo vae) |
| `--use_sample_images` | off | Descarga manzana y banana de Wikipedia |

---

## Estructura

```
flux_klein/
├── src/
│   ├── pipeline.py      # load_vae, interpolate_vae, load_klein_pipeline, interpolate_klein
│   └── visualize.py     # save_comparison, save_alpha_sweep
├── tests/
│   └── test_pipeline.py # 5 unit tests, sin GPU requerida
├── run.py               # Entry point CLI
├── requirements.txt
└── README.md
```

---

## Outputs generados

| Archivo | Descripción |
|---|---|
| `results/comparison_vae.png` | Apple \| Midpoint \| Banana (modo VAE) |
| `results/comparison_klein.png` | Apple \| Midpoint \| Banana (modo Klein) |
| `results/alpha_sweep.png` | Strip α∈{0, 0.25, 0.5, 0.75, 1.0} |
| `results/midpoint_*.png` | Imagen standalone del punto medio |

---

## Tests (sin GPU)

```bash
python tests/test_pipeline.py
```

```
=== FLUX.2-klein-4B Pipeline Tests ===

[PASS] test_preprocess
[PASS] test_alpha_math
[PASS] test_decode_pil
[PASS] test_save_comparison
[PASS] test_interpolate_vae_mock

✓ All 5 tests passed!
```

---

## Requisitos

- Python 3.10+
- GPU con ≥16 GB VRAM para modo `klein` (RTX 3090/4090, A10, T4 Colab)
- `diffusers` desde git (Flux2KleinPipeline aún no está en PyPI estable)
- `black-forest-labs/FLUX.2-klein-4B` en HuggingFace (Apache 2.0, libre)
