import os
import argparse
import shutil
import random
from glob import glob
import yaml
from roboflow import Roboflow
from dotenv import load_dotenv
from src.utils.logger import logger

load_dotenv()

def download_dataset(api_key, workspace, project, version):
    """
    Download dataset from Roboflow in YOLOv8 format.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8")
    return dataset.location

def stratified_split(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split a dataset into train/val/test sets.
    """
    random.seed(seed)

    # Supported image formats
    image_exts = ['*.jpg', '*.jpeg', '*.png']
    images = []
    for ext in image_exts:
        images.extend(glob(os.path.join(image_dir, ext)))
    images = sorted(images)

    data = []
    for img_path in images:
        base = os.path.basename(img_path).rsplit('.', 1)[0]
        lbl_path = os.path.join(label_dir, base + ".txt")
        if os.path.exists(lbl_path):
            data.append((img_path, lbl_path))

    if not data:
        logger.warning(f"No valid images and labels found in {image_dir} and {label_dir}")
        return

    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        'train': data[:n_train],
        'val': data[n_train:n_train + n_val],
        'test': data[n_train + n_val:]
    }

    for split_name, split_data in splits.items():
        split_img_dir = os.path.join(output_dir, split_name, 'images')
        split_lbl_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for img_path, lbl_path in split_data:
            shutil.copy(img_path, os.path.join(split_img_dir, os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(split_lbl_dir, os.path.basename(lbl_path)))

    logger.info(f"Dataset split successfully into: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

def main():
    parser = argparse.ArgumentParser(description="Prepare PCB Dataset")
    parser.add_argument("--config", type=str, help="Path to data_config.yaml")
    parser.add_argument("--download", action="store_true", help="Download from Roboflow")

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    if args.download:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            logger.error("ROBOFLOW_API_KEY not found in environment. Please set it in .env file.")
            return

        # Values from config or defaults
        workspace = config.get('roboflow', {}).get('workspace', 'arshiasbn-4s0xh')
        project_name = config.get('roboflow', {}).get('project', 'detecting-the-pcb-object')
        version = config.get('roboflow', {}).get('version', 3)

        logger.info(f"Downloading dataset {project_name} (v{version}) from Roboflow...")
        location = download_dataset(api_key, workspace, project_name, version)
        logger.info(f"Dataset downloaded to {location}")
    else:
        # Use local paths from config
        raw_path = config.get('raw_data_path', 'data/raw')
        output_path = config.get('path', 'data')

        img_dir = os.path.join(raw_path, 'images')
        lbl_dir = os.path.join(raw_path, 'labels')

        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            ratios = config.get('split_ratio', {'train': 0.7, 'val': 0.2, 'test': 0.1})
            seed = config.get('seed', 42)
            logger.info(f"Splitting local dataset from {raw_path} to {output_path}...")
            stratified_split(img_dir, lbl_dir, output_path,
                             ratios['train'], ratios['val'], ratios['test'], seed)
        else:
            logger.warning(f"Raw data not found at {img_dir} or {lbl_dir}. Nothing to split.")

if __name__ == "__main__":
    main()
