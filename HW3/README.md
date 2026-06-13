# Assignment 3 — Deep Learning

PyTorch implementations covering feedforward networks, CNNs, RNNs, and Transformers, with systematic ablation studies across optimizers, architectures, regularization, and transfer learning.

**Course:** Data Science · **Deliverable:** [`main.ipynb`](main.ipynb)

---

## Overview

| Part | Topic | Dataset | Task |
|------|-------|---------|------|
| **1** | Multilayer Perceptron (MLP) | House Prices ([Kaggle](https://www.kaggle.com/datasets/shree1992/housedata?select=data.csv)) | Regression + binary classification |
| **2** | Convolutional Neural Networks (CNN) | [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 4-class image classification |
| **3** | Recurrent Neural Networks (RNN) | [BBC News Archive](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive/data) | 5-class text classification |
| **4** | Transformers | BBC News (shared) | Native encoder + DistilBERT fine-tuning |
| **5** | Report | — | Submitted separately as PDF |

---

## Project Structure

```
HW3/
├── main.ipynb
├── part5_report.pdf
├── part-1-dataset/
│   └── data-part1.csv
├── part-2-dataset/                 # large — download separately if missing
│   ├── Training/
│   └── Testing/
└── part-3&4-dataset/
    └── bbc-news-data.csv
```

---

## Setup

**Requirements:** Python 3.10+, CUDA optional

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn transformers
```

### Datasets

1. **Part 1** — `data-part1.csv` is in `part-1-dataset/` (same Kaggle house-sales data as HW2).
2. **Part 2** — Download the Brain Tumor MRI dataset and extract to `part-2-dataset/` with `Training/` and `Testing/` subfolders.
3. **Parts 3 & 4** — `bbc-news-data.csv` is already included under `part-3&4-dataset/`.

---

## Running

Open and run [`main.ipynb`](main.ipynb) top to bottom. Each part is self-contained with markdown explanations, training loops, evaluation metrics, and plots.

GPU is used automatically when available (`cuda` → `cpu` fallback).

---

## What's Inside

### Part 1 — MLP

- `FlexibleMLP` — configurable depth, width, activations, batch norm, dropout, weight init
- `ModelTrainer` — unified training loop with early stopping, gradient clipping, metric logging
- **Tasks:** price regression (MSE) · expensive-house classification (BCEWithLogits)
- **Ablations:** optimizers (SGD / momentum / Adam) · learning rates · activations · batch norm · dropout & L2

### Part 2 — CNN

- Custom CNN baseline on 224×224 MRI scans (4 classes: glioma, meningioma, pituitary, no tumor)
- Component experiments: pooling, filters, depth
- Data augmentation (flip, rotation, color jitter)
- Transfer learning with **ResNet18** (ImageNet weights)
- Bonus: feature-map visualization via forward hooks

### Part 3 — RNN

- Unified `SequenceModel` supporting **RNN**, **LSTM**, and **GRU** (uni/bidirectional)
- Vocabulary encoding, padding/truncation to fixed sequence length
- Comparisons across recurrent architectures and hyperparameters

### Part 4 — Transformers

- Native `PyTorchNativeTransformer` with positional encoding and multi-head attention
- **DistilBERT** fine-tuning via Hugging Face `AutoTokenizer`
- Side-by-side benchmark: RNN vs LSTM vs GRU vs Transformer

### Part 5

Written report provided as a separate PDF.

---

## Notes

- Large image datasets are gitignored; download links are listed above.
- Part 1 reuses the house-sales dataset from Assignment 2.
- Discussion questions and result analyses are answered inline in the notebook.
