# FLUX.2-klein-4B Latent Interpolation Pipeline

> Encuentra el **punto medio** entre dos imágenes en el espacio latente.  
> Ejemplo: 🍎 Manzana + 🍌 Banana = 🍌🍎 híbrido Apple-Banana.

Modelo: [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)  
Licencia: Apache 2.0 | VRAM requerida: **~13 GB** (safe para 16 GB)

---

## ⚡ Quick Start — Google Colab (GPU T4/A10)

Abre un nuevo notebook en Colab, activa una **GPU T4**, y ejecuta:

```python
# 1. Instalar diffusers desde git (necesario para Flux2KleinPipeline)
!pip install -q git+https://github.com/huggingface/diffusers.git
!pip install -q torch torchvision accelerate transformers Pillow matplotlib

# 2. Clonar este repo
!git clone https://github.com/Serces19/flux-latent-interpolation.git
%cd flux-latent-interpolation

# 3. Tests rápidos (sin GPU, sin descargar el modelo)
!python tests/test_pipeline.py

# 4a. Modo VAE – rápido, solo carga el VAE (~300 MB)
!python run.py --mode vae --use_sample_images --sweep

# 4b. Modo Klein – pipeline completo FLUX.2-klein-4B (~13 GB VRAM)
!python run.py --mode klein --use_sample_images --steps 4

# 5. Ver resultados en Colab
from IPython.display import Image as IPImage, display
display(IPImage('results/comparison_vae.png'))
display(IPImage('results/comparison_klein.png'))
```

---

## Arquitectura

```
Modo VAE (rápido):
  🍎 Apple  ─► VAE Encode ─► Latent A ──┐
                                           ├─ lerp(α=0.5) ─► Decode ─► 🍌🍎
  🍌 Banana ─► VAE Encode ─► Latent B ──┘

Modo Klein (semántico):
  🍎 Apple  ─┐
               ├─ Flux2KleinPipeline (multi-reference + prompt) ─► 🍌🍎
  🍌 Banana ─┘
```

---

## Modos de operación

| Flag | Modo | VRAM | Tiempo | Descripción |
|---|---|---|---|---|
| `--mode vae` | VAE lerp | ~0.5 GB | <5s | Interpolación matemática en espacio latente |
| `--mode klein` | Klein full | ~13 GB | ~10-30s | Blending semántico con el transformer 4B |

---

## CLI completo

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
| `--steps` | `4` | Pasos de inferencia (solo mode=klein) |
| `--sweep` | off | Genera sweep α∈{0,0.25,0.5,0.75,1} (solo vae) |
| `--use_sample_images` | off | Descarga manzana y banana de Wikipedia |

---

## Estructura del proyecto

```
flux-latent-interpolation/
├── src/
│   ├── pipeline.py      # load_vae, interpolate_vae, load_klein_pipeline, interpolate_klein
│   └── visualize.py     # save_comparison, save_alpha_sweep
├── tests/
│   └── test_pipeline.py # 4 unit tests, sin GPU requerida
├── run.py               # Entry point CLI
├── requirements.txt
└── README.md
```

---

## Outputs

| Archivo | Descripción |
|---|---|
| `results/comparison_vae.png` | Apple \| Midpoint \| Banana (VAE mode) |
| `results/comparison_klein.png` | Apple \| Midpoint \| Banana (Klein mode) |
| `results/alpha_sweep.png` | Strip α∈{0, 0.25, 0.5, 0.75, 1.0} |
| `results/midpoint_*.png` | Imagen standalone del punto medio |

---

## Tests (sin GPU)

```bash
python tests/test_pipeline.py
```

```
=== Running FLUX Interpolation Tests ===
[PASS] test_preprocess_shape_and_range
[PASS] test_alpha_interpolation_math
[PASS] test_decode_output_is_valid_image
[PASS] test_save_comparison_creates_file

All tests passed!
```

---

## Requisitos

- Python 3.10+
- CUDA GPU con ≥16 GB VRAM para modo `klein`
- `diffusers` desde git (Flux2KleinPipeline aún no está en PyPI estable)
- Acceso a HuggingFace: `black-forest-labs/FLUX.2-klein-4B` (Apache 2.0, libre)
