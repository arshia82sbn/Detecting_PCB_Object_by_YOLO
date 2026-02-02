# PCB Defect Detection using YOLO

Professional YOLO-based PCB defect detection project designed for academic publication and production deployment.

## Features
- **Fine-tuning** on custom PCB datasets.
- **Config-driven** architecture (YAML).
- **Design Patterns**: Singleton Logger, Factory for Models/Datasets, Pipeline for Train/Deploy.
- **Mixed-precision training** and deterministic runs.
- **Advanced Augmentations** using Albumentations.
- **Deployment-ready** inference with JSON metadata output.
- **Export options** (ONNX, TorchScript).

## Project Structure
```
pcb_yolo/
├── configs/           # YAML configurations
├── data/              # Dataset splits (mock and real)
├── src/               # Core source code
│   ├── datasets/      # Data loading and factory
│   ├── models/        # YOLO wrapper, trainer, and factory
│   ├── utils/         # Logging, helpers, and metrics
│   ├── inference/     # Deployment detector
│   └── pipeline/      # Train and Deploy pipelines
├── scripts/           # CLI entry points
├── experiments/       # Logs, weights, and results
├── docs/              # Academic documentation
└── tests/             # Unit and integration tests
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r pcb_yolo/requirements.txt -r pcb_yolo/requirements-dev.txt
```

## Usage

### 1. Prepare Dataset
```bash
# Locally split raw data
python -m scripts.prepare_dataset --config configs/data_config.yaml

# Or download from Roboflow
python -m scripts.prepare_dataset --config configs/data_config.yaml --download
```

### 2. Training
```bash
python -m src.train --config configs/train_config.yaml --seed 42 --deterministic
```

### 3. Export
```bash
python -m src.export --model experiments/pcb_train/model.pt --format onnx
```

### 4. Deployment / Inference
```bash
python -m src.inference.detector --model experiments/pcb_train/model.pt --input data/mock/test/images --output experiments/test_predictions
```

## Testing
```bash
cd pcb_yolo
export PYTHONPATH=$PYTHONPATH:.
python -m pytest tests
```

## Academic Use
This project tracks metrics (mAP@0.5, Precision, Recall, F1) and saves publication-quality annotated images and detection metadata in JSON format for easy table generation. See `docs/experiment_protocol.md` for details.
