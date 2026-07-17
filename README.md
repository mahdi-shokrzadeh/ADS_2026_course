# Data Science — Course Assignments (2026)

Homework submissions for the DS course, organized by assignment in separate folders.

## Assignments

| Assignment | Topic | Folder |
|------------|-------|--------|
| **HW2** | House price regression & classification | [`HW2/`](HW2/) |
| **HW3** | Deep learning (MLP, CNN, RNN, Transformers) | [`HW3/`](HW3/) |
| **HW4** | Pipelines, imbalance, generative models & XAI | [`HW4/`](HW4/) |

---

### [HW2 — House Price Prediction & Classification](HW2/README.md)

End-to-end ML on real-estate sales data: EDA and preprocessing, then three tasks on `price` — continuous regression (Linear, Ridge, LASSO, Kernel Ridge), binary classification (expensive vs. not), and multiclass classification (four price quartiles). Includes theoretical write-ups on regularization, bias–variance, and evaluation metrics.

**Entry point:** [`HW2/main_HW2.ipynb`](HW2/main_HW2.ipynb)

---

### [HW3 — Deep Learning](HW3/README.md)

PyTorch implementations across four parts: **MLP** on house prices (regression + binary classification with optimizer/architecture ablations), **CNN** on brain-tumor MRI (custom CNN + ResNet18 transfer learning), **RNN** on BBC news (RNN/LSTM/GRU text classification), and **Transformers** (native encoder + DistilBERT fine-tuning). Written report submitted separately as PDF.

**Entry point:** [`HW3/main.ipynb`](HW3/main.ipynb)

---

### [HW4 — Generative Models, Pipelines & XAI](HW4/README.md)

Modern workflows: sklearn/pandas **pipelines** and imputation, **imbalanced learning** (undersample / oversample / SMOTE / class weights), generative models (**VAE**, **GAN**, **Diffusion**), and **XAI** (Grad-CAM, SHAP, LIME, ELI5) on the Assignment 3 CNN.

**Entry point:** [`HW4/assignement4.ipynb`](HW4/assignement4.ipynb)

---

## Repository layout

```
.
├── HW2/
│   ├── README.md
│   ├── main_HW2.ipynb
│   └── data.csv
├── HW3/
│   ├── README.md
│   ├── main.ipynb
│   ├── part-1-dataset/
│   ├── part-2-dataset/          # large — gitignored
│   ├── part-3&4-dataset/
│   └── part5_report.pdf
├── HW4/
│   ├── README.md
│   ├── assignement4.ipynb
│   └── Ass4-datasets/           # heavy subfolders gitignored
└── README.md
```

Add new assignments as `HWn/` with their own README and notebook(s).
