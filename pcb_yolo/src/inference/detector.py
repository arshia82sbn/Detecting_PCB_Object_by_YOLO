import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
from src.utils.logger import logger
from src.utils.helpers import save_json

class PCBDetector:
    """
    Inference pipeline for PCB defect detection with production-ready features.
    """
    def __init__(self, model_path, deploy_config):
        self.model = YOLO(model_path)
        self.config = deploy_config

    def detect(self, source, output_dir=None):
        """
        Run inference on the source and save results.
        """
        logger.info(f"Running inference on {source}")
        
        if output_dir and not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)

        results = self.model.predict(
            source=source,
            conf=self.config.get('conf_threshold', 0.25),
            iou=self.config.get('iou_threshold', 0.45),
            imgsz=self.config.get('imgsz', 640),
            device=self.config.get('device', 'cpu'),
            save=True if output_dir else False,
            project=output_dir if output_dir else None,
            name='predictions',
            exist_ok=True
        )

        detection_metadata = []
        for result in results:
            path = result.path
            # boxes.xywh is [x_center, y_center, width, height]
            # boxes.xyxy is [x1, y1, x2, y2]
            # Let's use x1, y1, w, h for the JSON as it's common for deployment
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            names = result.names

            meta = {
                "image": os.path.basename(path),
                "detections": []
            }
            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                w = x2 - x1
                h = y2 - y1
                meta["detections"].append({
                    "class": names[int(classes[i])],
                    "score": round(float(scores[i]), 4),
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(w), 2), round(float(h), 2)]
                })
            detection_metadata.append(meta)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_json(detection_metadata, os.path.join(output_dir, "detections.json"))
            logger.info(f"Detections saved to {output_dir}")

        return results, detection_metadata

def main():
    import argparse
    from src.utils.helpers import load_yaml

    parser = argparse.ArgumentParser(description="Run PCB YOLO Detector")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--input", type=str, required=True, help="Input source (image, dir, video)")
    parser.add_argument("--output", type=str, default="experiments/test_predictions", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/deploy_config.yaml", help="Path to deploy config")
    
    args = parser.parse_args()
    
    # Adjust paths if run from src/
    config_path = args.config
    if not os.path.exists(config_path):
        # try relative to root
        config_path = os.path.join(os.getcwd(), config_path)

    deploy_cfg = load_yaml(config_path)
    detector = PCBDetector(args.model, deploy_cfg)
    detector.detect(args.input, args.output)

if __name__ == "__main__":
    main()
