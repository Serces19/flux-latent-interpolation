"""
colab_demo.py  –  FLUX.2-klein-4B Interactive Demo

HOW TO USE in Google Colab:
  Run each section below as a SEPARATE notebook cell.
  Copy-paste the "=== CELL X ===" blocks into individual cells.

  DO NOT run this file with:  !python colab_demo.py
  Instead use:                %run colab_demo.py
  Or paste cells individually.
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies                          ║
# ╚══════════════════════════════════════════════════════════╝
# Paste this into Cell 1:
#
#   !pip install -q git+https://github.com/huggingface/diffusers.git
#   !pip install -q torch torchvision accelerate transformers Pillow matplotlib
#
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2 — Clone repo (safe, only once)                  ║
# ╚══════════════════════════════════════════════════════════╝
# Paste this into Cell 2:
#
#   import os
#   if not os.path.exists('/content/flux-latent-interpolation'):
#       !git clone https://github.com/Serces19/flux-latent-interpolation.git /content/flux-latent-interpolation
#   os.chdir('/content/flux-latent-interpolation')
#   print("Working directory:", os.getcwd())
#
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Upload Image A                                ║
# ╚══════════════════════════════════════════════════════════╝
# Paste this into Cell 3:
#
#   from google.colab import files
#   from pathlib import Path
#   from IPython.display import display, Image as IPImage
#
#   Path("assets").mkdir(exist_ok=True)
#   print("Upload your FIRST image (e.g. apple, cat, car...)")
#   up_a = files.upload()
#   fname_a = list(up_a.keys())[0]
#   path_a = f"assets/image_a_{fname_a}"
#   with open(path_a, "wb") as f:
#       f.write(up_a[fname_a])
#   print(f"✓ Image A saved: {path_a}")
#   display(IPImage(path_a, width=300))
#
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Upload Image B                                ║
# ╚══════════════════════════════════════════════════════════╝
# Paste this into Cell 4:
#
#   print("Upload your SECOND image (e.g. banana, dog, truck...)")
#   up_b = files.upload()
#   fname_b = list(up_b.keys())[0]
#   path_b = f"assets/image_b_{fname_b}"
#   with open(path_b, "wb") as f:
#       f.write(up_b[fname_b])
#   print(f"✓ Image B saved: {path_b}")
#   display(IPImage(path_b, width=300))
#
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5a — Run VAE interpolation (fast, ~300 MB)        ║
# ╚══════════════════════════════════════════════════════════╝
# Paste this into Cell 5:
#
#   from src.pipeline import load_vae, interpolate_vae
#   from src.visualize import save_comparison, save_alpha_sweep
#   from pathlib import Path
#   from IPython.display import display, Image as IPImage
#
#   ALPHA = 0.5   # ← change this: 0=image A, 0.5=midpoint, 1=image B
#   SWEEP = True  # ← set False to skip alpha sweep
#
#   Path("results").mkdir(exist_ok=True)
#   vae = load_vae(device="cuda")
#   img_a, img_b, img_mid = interpolate_vae(
#       vae, path_a, path_b, alpha=ALPHA, size=512, device="cuda"
#   )
#   label_a = Path(path_a).stem.split("_", 2)[-1]
#   label_b = Path(path_b).stem.split("_", 2)[-1]
#
#   img_mid.save(f"results/midpoint_vae_alpha{ALPHA}.png")
#   save_comparison(img_a, img_b, img_mid,
#                   label_a=label_a, label_b=label_b,
#                   label_mid=f"{label_a}-{label_b} α={ALPHA}",
#                   output_path="results/comparison_vae.png")
#   display(IPImage("results/comparison_vae.png"))
#
#   if SWEEP:
#       save_alpha_sweep(vae, path_a, path_b,
#                        alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
#                        size=512, device="cuda",
#                        output_path="results/alpha_sweep.png")
#       display(IPImage("results/alpha_sweep.png"))
#
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5b — Run Klein full pipeline (~13 GB VRAM)        ║
# ╚══════════════════════════════════════════════════════════╝
# Paste this into Cell 5 INSTEAD of 5a (choose one):
#
#   from src.pipeline import load_klein_pipeline, interpolate_klein
#   from src.visualize import save_comparison
#   from pathlib import Path
#   from IPython.display import display, Image as IPImage
#
#   ALPHA = 0.5   # ← change this
#   STEPS = 4     # ← more steps = better quality but slower
#   SEED  = 42
#
#   Path("results").mkdir(exist_ok=True)
#   pipe = load_klein_pipeline(device="cuda")
#   img_a, img_b, img_mid = interpolate_klein(
#       pipe, path_a, path_b,
#       alpha=ALPHA, size=1024, steps=STEPS, seed=SEED
#   )
#   label_a = Path(path_a).stem.split("_", 2)[-1]
#   label_b = Path(path_b).stem.split("_", 2)[-1]
#
#   img_mid.save(f"results/midpoint_klein_alpha{ALPHA}.png")
#   save_comparison(img_a, img_b, img_mid,
#                   label_a=label_a, label_b=label_b,
#                   label_mid=f"{label_a}-{label_b} α={ALPHA} [klein]",
#                   output_path="results/comparison_klein.png")
#   display(IPImage("results/comparison_klein.png"))
