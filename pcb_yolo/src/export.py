import argparse
import os
from ultralytics import YOLO
from src.utils.logger import logger

def export_model(model_path, format='onnx'):
    """
    Export the YOLO model to the specified format (ONNX, TorchScript, etc.).
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None

    logger.info(f"Loading model from {model_path} for export to {format}...")
    model = YOLO(model_path)
    
    # Export the model
    # imgsz can be specified if needed, defaults to 640
    try:
        exported_path = model.export(format=format)
        logger.info(f"Model exported successfully to {exported_path}")
        return exported_path
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "openvino", "engine"], help="Export format")
    
    args = parser.parse_args()
    export_model(args.model, args.format)

if __name__ == "__main__":
    main()
