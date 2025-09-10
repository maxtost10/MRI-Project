# Cascaded Deep Learning for MRI Reconstruction (Synthetic Phantoms)

**TL;DR.** I reconstruct home-made phantom images after artificial undersampling.  
I first compared a **TV-denoising baseline** against my **U-Net that reconstructs in k-space**.  
Surprisingly, **TV held up in SSIM**. That led to a new **cascaded design**: *reconstruct in k-space → inverse FFT → refine in image space*.  
The cascaded model yields **clearly better PSNR/SSIM and lowest RMSE**. See figures and samples below.

---

## At a glance

- **Data**: synthetic Shepp–Logan–style phantoms, 128×128, R=4 undersampling with ACS.
- **Baselines**: zero-filled IFFT, TV-denoise (ROF model), k-space U-Net with data consistency.
- **New idea**: **stack CNNs** → Stage 1: k-space U-Net (frozen) → Stage 2: image-space refinement CNN (residual).
- **Metrics**: PSNR, SSIM, RMSE — all computed after **consistent GT-based normalization** to avoid metric inflation.
- **Outcome**: Cascaded model > U-Net > TV ≈ zero-filled (SSIM can flatter TV); cascaded wins across metrics.

---

## Why this was interesting

1. **Starting point** – Reconstruct undersampled phantoms. A U-Net in frequency space already does well thanks to **data consistency** (we hard-replace predicted k-space with measured samples under the mask).
2. **Surprise** – On **SSIM**, a simple **TV-denoiser** occasionally matched the U-Net. TV's piecewise-smooth prior can score well on SSIM even while losing fine detail.
3. **Idea** – If TV's strength is structural smoothing and the U-Net's strength is data-consistent infill, why not **compose** them? I built a **cascaded model**: predict k-space with the U-Net, IFFT to an image, then **refine the image with a second CNN**.
4. **Result** – The cascaded model **preserves structure and edges** while reducing residual artifacts, giving **higher PSNR/SSIM and the lowest RMSE**.

---

## Figures

### 1) Baseline comparison — Zero-filled vs U-Net vs TV-Denoise
*(20 test phantoms; PSNR/SSIM/RMSE distributions)*

![Reconstruction Performance — Baselines](./Plots/performance_comparison_tv_denoise.png)

**Observation.** U-Net beats zero-filled on all metrics. **TV-denoise can look competitive on SSIM** because smoothing boosts local structural similarity; however, TV typically underperforms on PSNR and RMSE and **removes fine detail**.

---

### 2) Cascaded approach — K-space U-Net → Image-space CNN
*(20 test phantoms; PSNR/SSIM/RMSE distributions)*

![Reconstruction Performance — Cascaded](./Plots/performance_comparison_cascaded.png)

**Observation.** The **cascaded model dominates**: highest PSNR/SSIM, lowest RMSE, and tighter spread (more stable across phantoms).

---

## Qualitative samples

**Baselines (zero-filled / U-Net / TV)** — per-sample panels:

![TV Denoise Comparison Sample 0](./Plots/comparison_tv_denoise_sample_0.png)
![TV Denoise Comparison Sample 1](./Plots/comparison_tv_denoise_sample_1.png)
![TV Denoise Comparison Sample 2](./Plots/comparison_tv_denoise_sample_2.png)

**Cascaded vs others** — per-sample panels:

![Cascaded Comparison Sample 0](./Plots/comparison_cascaded_sample_0.png)
![Cascaded Comparison Sample 1](./Plots/comparison_cascaded_sample_1.png)
![Cascaded Comparison Sample 2](./Plots/comparison_cascaded_sample_2.png)

These show images, error maps, and (for cascaded) difference maps vs the other methods.

---

## Methods (short)

- **Zero-filled**: inverse FFT of undersampled k-space.
- **TV-Denoise**: Rudin–Osher–Fatemi (ROF) model  
  ```
  min_u (1/2)||u-x||_2^2 + λ·TV(u)
  ```
  implemented via a fast dual update with projection onto the unit ball.
- **K-space U-Net**: adaptive-kernel U-Net operating on **real/imag channels in k-space**, with **built-in data consistency** (replace predicted values where mask==1).
- **Cascaded model**: **Stage 1** (frozen) k-space U-Net → **IFFT** → **Stage 2** image-space refinement CNN (residual learning). Loss combines L1/L2; light L1 weight decay on refinement.

---

## Metrics & evaluation hygiene

All metrics are computed after **normalizing both reconstruction and prediction w.r.t. the ground-truth intensity range**:
- Avoids the pitfall of independently normalizing each image to [0,1], which can **inflate PSNR/SSIM** for overly smooth outputs.
- We report **PSNR, SSIM, RMSE** on a held-out test set (n=20).

---

## Why the cascade helps

- **K-space stage**: excels at **data-consistent infill** and respects measured Fourier coefficients.
- **Image stage**: exploits **spatial priors** (edges, shapes) to **de-alias** and **sharpen**.
- The composition reduces artifacts without the over-smoothing typical of TV, giving **better fidelity and perceptual quality**.

---

## How to reproduce (scripts)

- **Train k-space U-Net**: `train_unet.py`  
- **Compare baselines** (zero-filled / U-Net / TV): `compare_methods.py`  
- **Train cascaded refinement** (k-space U-Net frozen): `train_cascaded_refinement.py`  
- **Compare cascaded vs baselines**: `compare_cascased_with_tv.py`

Expected inputs:
- `mri_dataset.h5` with groups `train/`, `test/` containing:  
  `phantoms`, `kspace_full`, `kspace_undersampled`, `masks` (R=4 with ACS).
- Saved U-Net weights `adaptive-unet.pth` and cascaded refinement `refinement_model.pth`.

> **Note.** Utility functions (k-space↔image, complex↔tensor, metrics, TV-denoise) live in a shared `utils.py` and are imported by the comparison scripts.

---

## Repository structure

```
├── train_unet.py                    # Adaptive U-Net (k-space) + Lightning training
├── train_cascaded_refinement.py     # Stage-2 image refinement + Lightning training
├── compare_methods.py               # Zero-filled vs U-Net vs TV baseline study
├── compare_cascased_with_tv.py      # Cascaded vs K-space U-Net vs TV
├── utils.py                         # Shared helpers (complex↔tensor, FFTs, metrics, TV)
├── mri_dataset.h5                   # Synthetic dataset (phantoms, masks, k-space)
└── Plots/                           # Figures used in this README
    ├── performance_comparison_tv_denoise.png
    ├── performance_comparison_cascaded.png
    ├── comparison_tv_denoise_sample_*.png
    └── comparison_cascaded_sample_*.png
```

---

## Technical foundations

This work builds on comprehensive exploration of MRI reconstruction fundamentals developed in this repository:

### 📁 **Classical Reconstruction Methods** (`src/ImageReconstruction/`)

- **[First Reconstruction](src/ImageReconstruction/first_reconstruction/)**: Zero-filled reconstruction and k-space fundamentals
- **[Interpolation Methods](src/ImageReconstruction/different_interpolation_methods/)**: Linear/spline k-space interpolation, radial interpolation, and low-pass filtering analysis

These foundational studies revealed the limitations of classical approaches and motivated the deep learning methodology presented here.

---

## What I'd explore next

- **Multi-coil data** and SENSE-like data consistency.
- **Non-Cartesian sampling** and graph/spiral trajectories.
- **Joint learning of sampling + reconstruction**, and physics-informed DC layers.
- **Perceptual/SSIM-aware losses** balanced against fidelity losses.
- **Uncertainty** estimates (e.g., MC dropout) for downstream decision support.

---

## Technical Stack
- **Deep Learning**: PyTorch, PyTorch Lightning
- **Scientific Computing**: NumPy, SciPy, scikit-image
- **Data & Logging**: HDF5, Weights & Biases
- **Visualization**: Matplotlib, PIL

---

If you have questions or would like to see this run on your data (multi-coil or non-Cartesian), I'm happy to adapt the pipeline.