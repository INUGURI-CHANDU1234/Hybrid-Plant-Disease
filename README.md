# 🌿 Hybrid Plant Disease Detection using RAG & EfficientNet-B4

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An advanced deep learning and Retrieval-Augmented Generation (RAG) framework for high-accuracy plant disease classification and automated treatment recommendations.

---

## 📌 Project Overview

Plant diseases severely threaten global crop yields and agricultural sustainability. This repository contains the complete computer vision classification pipeline built using **PyTorch**, fine-tuned **EfficientNet-B4**, and custom **Multi-class Focal Loss** on the **PlantVillage** dataset (~20,639 images across 15 plant disease classes).

### Key Features
- **Deep Learning Backbone:** EfficientNet-B4 pretrained on ImageNet ($380 \times 380$ input resolution).
- **Custom Loss Function:** Multi-class Focal Loss ($\gamma=2.0$, $\alpha=1.0$) to mitigate class imbalance and focus on hard examples.
- **Advanced Augmentations:** Powered by `Albumentations` (`RandomResizedCrop`, `HorizontalFlip`, `Rotate`, `ColorJitter`).
- **Robust Data Pipeline:** Auto-detects and gracefully skips corrupted/damaged image files without interrupting training execution.
- **Publication-Ready Visualizations:** Automated confusion matrix, probability correlation heatmap, and high-DPI loss/accuracy curves.
- **RAG Architecture Integration:** Designed to pair vision predictions with Vector Databases & LLMs for intelligent treatment synthesis.

---

## 📊 Performance Metrics

Evaluated on an 80/20 stratified validation split:

| Metric | Performance |
| :--- | :--- |
| **Accuracy** | **99.61%** |
| **Macro F1-Score** | **0.9961** |
| **Macro Precision** | **0.9966** |
| **Macro Recall** | **0.9956** |

---

## 📁 Repository Structure

```
├── hybrid-plant-disease-detection-using-rag.ipynb   # Main PyTorch Training & Evaluation Notebook
├── README.md                                         # Project Documentation
├── .gitignore                                        # Git Ignore Rules
└── requirements.txt                                  # Environment Dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install required Python packages:

```bash
git clone https://github.com/INUGURI-CHANDU1234/Hybrid-Plant-Disease.git
cd Hybrid-Plant-Disease
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the **PlantVillage** dataset from Kaggle or your preferred source and update `cfg.data_dir` inside the notebook:

```python
@dataclass
class CFG:
    data_dir: str = "/path/to/PlantVillage"
    img_size: int = 380
    batch_size: int = 8
    epochs: int = 15
    lr: float = 1e-4
```

### 3. Run Notebook
Launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook hybrid-plant-disease-detection-using-rag.ipynb
```

---

## ⚙️ Model Architecture & Training Config

```python
# EfficientNet-B4 + Custom Classification Head
model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, num_classes)
)
```

- **Optimizer:** `AdamW` (learning rate: `1e-4`, weight decay: `1e-4`)
- **Learning Rate Scheduler:** `CosineAnnealingLR` ($T_{\max}=15$)
- **Mixed Precision:** PyTorch `autocast` + `GradScaler`

---

## 🗺️ Future Roadmap

- [ ] **Vector Database Indexing:** Store plant treatment guides and agronomic literature in FAISS/ChromaDB.
- [ ] **RAG Treatment Synthesizer:** Connect vision model outputs with an LLM (e.g., Gemini / LLaMA) for automated dosage & prevention recommendations.
- [ ] **Web Dashboard:** Deploy interactive React + FastAPI web application for real-time leaf image upload and diagnosis.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
