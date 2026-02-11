import argparse
import os
import random
import numpy as np
import torch
from src.pipeline.train_pipeline import TrainPipeline
from src.utils.logger import logger
from src.utils.helpers import load_yaml

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Train PCB YOLO Model")
    parser.add_argument("--config", type=str, required=True, help="Path to train config")
    parser.add_argument("--data", type=str, default="configs/data_config.yaml", help="Path to data config")
    parser.add_argument("--model", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--ci", action="store_true", help="CI mode (short run)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic training")
    
    args = parser.parse_args()

    set_global_seeds(args.seed)

    pipeline = TrainPipeline(args.data, args.model, args.config)
    
    # Overrides from CLI
    pipeline.train_cfg['seed'] = args.seed
    if args.deterministic:
        pipeline.train_cfg['deterministic'] = True

    try:
        results = pipeline.run()
        logger.info("Training Finished Successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    main()
