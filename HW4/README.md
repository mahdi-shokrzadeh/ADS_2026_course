# Assignment 4 — Generative Models, Pipelines & XAI

Modern DS workflows in one notebook: ML pipelines, imbalanced learning, generative models (VAE / GAN / Diffusion), and explainable AI.

**Course:** Data Science · **Deliverable:** [`assignement4.ipynb`](assignement4.ipynb)

---

## Overview

| Part | Topic | Dataset |
|------|-------|---------|
| **1** | ML pipelines (`pandas.pipe`, `sklearn.pipeline`, imputation) | House prices |
| **2** | Imbalanced data (undersample / oversample / SMOTE / class weights) | House prices (`luxury` target) |
| **3** | Variational Autoencoders (β-VAE, latent interpolation) | FashionMNIST |
| **4** | GANs (generator, discriminator, training loop) | MNIST |
| **5** | Diffusion (DDPM; linear vs cosine schedule) | MNIST |
| **6** | XAI (Grad-CAM, SHAP, LIME, ELI5) | Brain Tumor MRI + CNN |

---

## Structure

```
HW4/
├── assignement4.ipynb
├── README.md
└── Ass4-datasets/
    ├── house-data.csv           # included
    ├── brain-tumor-mri/         # gitignored — download separately
    ├── torch-data/              # gitignored — auto-downloaded by torchvision
    └── artifacts/               # gitignored — saved models / samples
```

---

## Setup

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn \
  imbalanced-learn joblib shap lime eli5
```

### Datasets

1. **House data** — `Ass4-datasets/house-data.csv` (same Kaggle set as HW2/HW3).
2. **MNIST / FashionMNIST** — downloaded automatically into `Ass4-datasets/torch-data/`.
3. **Brain Tumor MRI** — same Kaggle set as HW3; place under `Ass4-datasets/brain-tumor-mri/` with `Training/` and `Testing/`.

---

## Running

Open [`assignement4.ipynb`](assignement4.ipynb) and run top to bottom. Parts build on each other (pipelines → imbalance → generative → XAI). GPU is used when available.
