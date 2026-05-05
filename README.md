# FLUX.2-klein-4B Latent Interpolation Pipeline

> Encuentra el **punto medio en espacio latente** entre dos imágenes que TÚ subes.
> Ejemplo: 🍎 Manzana + 🍌 Banana = 🍌🍎 Apple-Banana híbrido.

**Modelo**: [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
**Licencia**: Apache 2.0 | **VRAM**: ~13 GB (seguro en 16 GB)

---

## ⚡ Google Colab — Celdas exactas para pegar

> **IMPORTANTE**: Copia cada bloque en una celda **separada** del notebook.
> NO uses `!python run.py` para el upload — `files.upload()` solo funciona dentro del kernel de Colab.

---

### 📦 Celda 1 — Instalar dependencias

```python
!pip install -q git+https://github.com/huggingface/diffusers.git
!pip install -q torch torchvision accelerate transformers Pillow matplotlib
```

---

### 📂 Celda 2 — Clonar repo (solo una vez, sin paths anidados)

```python
import os

if not os.path.exists('/content/flux-latent-interpolation'):
    !git clone https://github.com/Serces19/flux-latent-interpolation.git /content/flux-latent-interpolation

os.chdir('/content/flux-latent-interpolation')
print("Working dir:", os.getcwd())
```

---

### 🖼️ Celda 3 — Subir Imagen A (tu primera imagen)

```python
from google.colab import files
from pathlib import Path
from IPython.display import display, Image as IPImage

Path("assets").mkdir(exist_ok=True)

print("Sube tu PRIMERA imagen (manzana, gato, auto, lo que quieras):")
up_a = files.upload()
fname_a = list(up_a.keys())[0]
path_a = f"assets/image_a_{fname_a}"
with open(path_a, "wb") as f:
    f.write(up_a[fname_a])
print(f"✓ Imagen A guardada: {path_a}")
display(IPImage(path_a, width=300))
```

---

### 🖼️ Celda 4 — Subir Imagen B (tu segunda imagen)

```python
print("Sube tu SEGUNDA imagen (banana, perro, camión, lo que quieras):")
up_b = files.upload()
fname_b = list(up_b.keys())[0]
path_b = f"assets/image_b_{fname_b}"
with open(path_b, "wb") as f:
    f.write(up_b[fname_b])
print(f"✓ Imagen B guardada: {path_b}")
display(IPImage(path_b, width=300))
```

---

### ⚡ Celda 5a — Modo VAE (rápido, ~300 MB, recomendado para empezar)

```python
from src.pipeline import load_vae, interpolate_vae
from src.visualize import save_comparison, save_alpha_sweep
from pathlib import Path
from IPython.display import display, Image as IPImage

ALPHA = 0.5   # 0=imagen A pura, 0.5=punto medio, 1=imagen B pura
SWEEP = True  # True = genera strip con α∈{0, 0.25, 0.5, 0.75, 1}

Path("results").mkdir(exist_ok=True)

vae = load_vae(device="cuda")
img_a, img_b, img_mid = interpolate_vae(
    vae, path_a, path_b, alpha=ALPHA, size=512, device="cuda"
)

label_a = Path(path_a).stem.split("_", 2)[-1]
label_b = Path(path_b).stem.split("_", 2)[-1]

img_mid.save(f"results/midpoint_vae_alpha{ALPHA}.png")
save_comparison(img_a, img_b, img_mid,
                label_a=label_a, label_b=label_b,
                label_mid=f"{label_a}-{label_b}  α={ALPHA}",
                output_path="results/comparison_vae.png")

print("── Resultado: A | Punto Medio | B ──")
display(IPImage("results/comparison_vae.png"))

if SWEEP:
    save_alpha_sweep(vae, path_a, path_b,
                     alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
                     size=512, device="cuda",
                     output_path="results/alpha_sweep.png")
    print("── Alpha Sweep ──")
    display(IPImage("results/alpha_sweep.png"))
```

---

### 🔥 Celda 5b — Modo Klein completo (~13 GB VRAM, semántico)

> Usa esta celda **en lugar de** la 5a si quieres el resultado con el transformer 4B.

```python
from src.pipeline import load_klein_pipeline, interpolate_klein
from src.visualize import save_comparison
from pathlib import Path
from IPython.display import display, Image as IPImage

ALPHA = 0.5   # 0=imagen A pura, 0.5=punto medio, 1=imagen B pura
STEPS = 4     # más pasos = mejor calidad pero más lento
SEED  = 42

Path("results").mkdir(exist_ok=True)

pipe = load_klein_pipeline(device="cuda")
img_a, img_b, img_mid = interpolate_klein(
    pipe, path_a, path_b,
    alpha=ALPHA, size=1024, steps=STEPS, seed=SEED
)

label_a = Path(path_a).stem.split("_", 2)[-1]
label_b = Path(path_b).stem.split("_", 2)[-1]

img_mid.save(f"results/midpoint_klein_alpha{ALPHA}.png")
save_comparison(img_a, img_b, img_mid,
                label_a=label_a, label_b=label_b,
                label_mid=f"{label_a}-{label_b}  α={ALPHA} [klein]",
                output_path="results/comparison_klein.png")

print("── Resultado Klein: A | Punto Medio | B ──")
display(IPImage("results/comparison_klein.png"))
```

---

## Tests (sin GPU, sin modelo)

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

## Arquitectura

```
Modo VAE (rápido, determinístico):
  Imagen A → VAE Encode → Latent A ──┐
                                      ├─ lerp(α) → Decode → Resultado
  Imagen B → VAE Encode → Latent B ──┘

Modo Klein (semántico, FLUX transformer 4B):
  Imagen A ──┐
              ├─ Flux2KleinPipeline (multi-reference + prompt) → Resultado
  Imagen B ──┘
```

---

## Modos

| Modo | VRAM | Tiempo | Descripción |
|---|---|---|---|
| VAE | ~0.5 GB | <5 s | Interpolación matemática directa en latentes |
| Klein | ~13 GB | ~10-30 s | Blending semántico con transformer FLUX 4B |

---

## Estructura

```
flux-latent-interpolation/
├── src/
│   ├── pipeline.py      # load_vae, interpolate_vae, load_klein_pipeline, interpolate_klein
│   └── visualize.py     # save_comparison, save_alpha_sweep
├── tests/
│   └── test_pipeline.py # 5 unit tests, sin GPU requerida
├── colab_demo.py        # guía de celdas con comentarios
├── run.py               # CLI para uso local
└── README.md
```
