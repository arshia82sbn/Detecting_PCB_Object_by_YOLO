import argparse
import os
from src.pipeline.deploy_pipeline import DeployPipeline
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Deploy PCB YOLO Model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image/video or directory")
    parser.add_argument("--output", type=str, default="experiments/predictions", help="Path to output directory")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--config", type=str, default="configs/deploy_config.yaml", help="Path to deploy config")
    parser.add_argument("--export", type=str, choices=['onnx', 'torchscript'], help="Export model format")

    args = parser.parse_args()

    pipeline = DeployPipeline(args.model, args.config)

    if args.export:
        pipeline.export_model(format=args.export)

    results, metadata = pipeline.run_inference(args.input, args.output)
    logger.info(f"Inference complete. Results saved in {args.output}")

if __name__ == "__main__":
    main()
