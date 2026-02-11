# Experiment Protocol

This document describes the reproduction protocol for the experiments reported in the academic article.

## Environment Setup
- Python 3.10+
- CUDA-enabled GPU (optional but recommended for speed)
- Dependencies listed in `requirements.txt`

## Hyperparameters
The following hyperparameters were selected for the fine-tuning stage:
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (init)
- **Batch Size**: 16
- **Epochs**: 100
- **Image Size**: 640x640
- **Mixed Precision**: Enabled (AMP)
- **Early Stopping**: 20 epochs patience

## Augmentation Recipe
Using Albumentations library:
- RandomRotate90
- Flip (Horizontal and Vertical)
- ShiftScaleRotate (p=0.5)
- RandomBrightnessContrast (p=0.2)
- Blur and Gaussian Noise (p=0.1)

## Reproducibility
- **Fixed Seed**: 42 (used for numpy, torch, and random)
- **Deterministic Flag**: Enabled for cuDNN operations.

## Metrics for Reporting
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- BoxF1 Score

Full curves (PR, F1) are saved in the experiment run directory under `experiments/<run_name>/`.
