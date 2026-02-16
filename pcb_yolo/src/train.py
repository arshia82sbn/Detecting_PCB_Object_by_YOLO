import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.utils.logger import logger


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_config_path(path: str) -> str:
    """Resolve config paths from either repo root or package root.

    This supports running the CLI from the repository root where configs live
    under ``pcb_yolo/configs`` as well as from inside the ``pcb_yolo`` folder.
    """
    input_path = Path(path)
    if input_path.exists():
        return str(input_path)

    package_root = Path(__file__).resolve().parents[1]
    fallback_path = package_root / input_path
    if fallback_path.exists():
        return str(fallback_path)

    return path


def main():
    parser = argparse.ArgumentParser(description="Train PCB YOLO Model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to train config"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config",
    )
    parser.add_argument("--ci", action="store_true", help="CI mode (short run)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic", action="store_true", help="Force deterministic training"
    )

    args = parser.parse_args()

    args.config = resolve_config_path(args.config)
    args.data = resolve_config_path(args.data)
    args.model = resolve_config_path(args.model)

    set_global_seeds(args.seed)

    # Defer heavy imports until CLI args are parsed so `--help` works even when
    # optional runtime deps (e.g., OpenCV system libs) are unavailable.
    from src.pipeline.train_pipeline import TrainPipeline

    pipeline = TrainPipeline(args.data, args.model, args.config)

    # Overrides from CLI
    pipeline.train_cfg["seed"] = args.seed
    if args.deterministic:
        pipeline.train_cfg["deterministic"] = True

    try:
        results = pipeline.run()
        logger.info("Training Finished Successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()
