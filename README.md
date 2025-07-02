# Glioma-CellCounter


This project implements a complete pipeline for semantic segmentation of medical images, focusing on distinguishing two types of cellular structures (A and B) in microscopy images. It leverages a deep learning architecture based on Nested U-Net (UNet++), implemented using PyTorch.

The system includes data preprocessing, training, evaluation, visualization, and performance reporting (including ROC/AUC analysis and object counting).

---

## 📁 Project Structure

```
├── train.py                    # Training script
├── test.py                     # Inference and evaluation script
├── util.py                     # Model architecture, dataset definitions, metrics & utilities
├── model.pth                   # Trained model weights (saved after training)
├── final_result_labeled/       # Visualizations and statistics on labeled test set
├── final_result_unlabeled/     # Results on unlabeled data (if provided)
├── auc_results/                # ROC curve CSV files
└── data/
    ├── train_images/           # Input images for training
    ├── train_labels/           # Corresponding label masks
    ├── test_images/            # Input images for testing
    ├── test_labels/            # Corresponding label masks
    └── unlabeled_images/       # (Optional) images without labels
```

---

## 🚀 Getting Started

### 1. Train the Model

```bash
python train.py
```

- Uses data from `data/train_images` and `data/train_labels`.
- Saves trained model to `model.pth`.
- Evaluates on test set and generates:
  - Prediction visualizations and object counts (`final_result_labeled/`)
  - Training ROC curves and per-class AUCs (`auc_results/`)

### 2. Run Inference

```bash
python test.py
```

- Loads `model.pth`.
- Evaluates the test set at `data/test_images`.
- Generates:
  - Macro recall bar chart for A and B classes
  - Combined ROC curve comparing train and test sets
  - If `data/unlabeled_images/` exists, processes unlabeled images and saves results to `final_result_unlabeled/`



---

## 📊 Output & Visualization

- Training Loss and Accuracy Curves: `final_result_labeled/training_loss.png`, `final_result_labeled/training_accuracy.png`
- Visual Results: Input, Ground Truth, and Prediction (side-by-side)
- Object Count Plot: `final_result_labeled/cell_count_barplot.pdf`
- Macro Recall per Image: `final_result_labeled/macro_recall_barplot.png`
- ROC Curve: `final_result_labeled/roc_combined.pdf`

---

## ⚙️ Dependencies

Install the following packages before running:

```bash
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
opencv-python-headless==4.11.0.86
numpy==1.23.5
matplotlib==3.7.2
pillow==11.0.0
scikit-learn==1.2.2
pandas==2.0.3
scipy==1.15.2
tqdm==4.67.1
albumentations==1.3.1
imageio==2.37.0
scikit-image==0.25.2
```

