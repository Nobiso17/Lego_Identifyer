# 🧱 LEGO Brick Classifier

A convolutional neural network (CNN) that identifies LEGO brick types from images, with support for real-time webcam inference. Built using transfer learning on MobileNetV2.

---

## Overview

This project trains an image classifier to recognize LEGO brick part numbers from photos. The pipeline covers everything from dataset exploration to live camera inference, making it a complete end-to-end ML project.

**Key features:**
- Transfer learning with MobileNetV2 (pre-trained on ImageNet)
- Two-phase training: feature extraction → fine-tuning
- Data augmentation to improve generalization
- Real-time webcam inference with confidence overlay
- Model export in Keras and TensorFlow SavedModel formats

---

## Project Structure

```
lego-brick-classifier/
├── lego_brick_classifier.ipynb   # Main notebook (full pipeline)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── models/                       # Saved model outputs (created at runtime)
    ├── best_lego_model.keras
    ├── best_lego_model_finetuned.keras
    ├── lego_saved_model/
    └── class_labels.json
```

---

## Dataset

The notebook expects a dataset folder where each subdirectory is a class (brick type) containing images:

```
dataset/
├── brick corner 1x2x2/
│   ├── 2357 brick corner 1x2x2 000L.png
│   └── 2357 brick corner 1x2x2 000R.png
├── brick 2x4/
│   └── 3001 brick 2x4 000L.png
└── ...
```

The class label is derived from the folder name. Images are `.png`.

> **Dataset used:** [Classify Bricks: Compare Transfer Learning Model](https://www.kaggle.com/code/databeru/classify-bricks-compare-transfer-learning-model/input?select=dataset). Transformed for each class to be in its own folder and stored in Google Drive. Pulled into Colab through `ZIP_PATH` in the config section.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/lego-brick-classifier.git
cd lego-brick-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Run locally:
```bash
jupyter notebook lego_brick_classifier.ipynb
```

Or open in **Google Colab** (recommended — free GPU access):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/lego-brick-classifier/blob/main/lego_brick_classifier.ipynb)

### 4. Configure paths

In **Section 2 (Configuration)**, update these variables to match your setup:

```python
ZIP_PATH     = "/content/drive/MyDrive/lego_dataset.zip"
ROOT_EXTRACT = "/content/lego_dataset"
DATASET_DIR  = "/content/lego_dataset/lego_dataset"
MODEL_DIR    = "/content/lego_dataset/models"
```

---

## Pipeline Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Install & Import | Dependencies and library imports |
| 2 | Configuration | All hyperparameters in one place |
| 3 | Dataset Exploration | Class counts, balance check, sample preview |
| 4 | Preprocessing & Augmentation | Resize, normalize, augment training images |
| 5 | Model Architecture | MobileNetV2 base + custom classification head |
| 6 | Phase 1 Training | Train head only (base frozen) |
| 7 | Phase 2 Fine-Tuning | Unfreeze top 30 layers, retrain with low LR |
| 8 | Training Plots | Accuracy & loss curves across both phases |
| 9 | Evaluation | Classification report + confusion matrix |
| 10 | Export | Save `.keras` and `SavedModel` formats |
| 11 | Single Image Inference | Test on any image file |
| 12 | Live Camera Inference | Real-time webcam classification |
| 13 | Load & Resume | Reload saved model without retraining |

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | `(224, 224)` | Input image dimensions |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS_FROZEN` | `15` | Phase 1 epochs (head only) |
| `EPOCHS_FINE` | `20` | Phase 2 epochs (fine-tuning) |
| `LEARNING_RATE` | `1e-3` | Phase 1 learning rate |
| `FINE_TUNE_LR` | `1e-5` | Phase 2 learning rate |
| `VALIDATION_SPLIT` | `0.2` | Fraction of data held out for validation |

---

## Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras** — model training and export
- **MobileNetV2** — pre-trained backbone (ImageNet weights)
- **OpenCV** — image loading and webcam inference
- **scikit-learn** — evaluation metrics and confusion matrix
- **Matplotlib / Seaborn** — training plots and visualizations
- **NumPy** — numerical operations

---

## License

MIT
