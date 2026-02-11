import argparse
import os
from src.pipeline.train_pipeline import TrainPipeline
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Train PCB YOLO Model")
    parser.add_argument("--data", type=str, default="configs/data_config.yaml", help="Path to data config")
    parser.add_argument("--model", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--train", type=str, default="configs/train_config.yaml", help="Path to train config")

    args = parser.parse_args()

    pipeline = TrainPipeline(args.data, args.model, args.train)
    try:
        results = pipeline.run()
        logger.info("Training Finished Successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
