from src.inference.detector import PCBDetector
from src.utils.helpers import load_yaml
from src.utils.logger import logger
import os


class DeployPipeline:
    """
    Orchestrates the deployment/inference process.
    """

    def __init__(self, model_path, deploy_cfg_path):
        self.model_path = model_path
        self.deploy_cfg = load_yaml(deploy_cfg_path)

    def run_inference(self, input_source, output_dir):
        logger.info(f"Initializing Deployment Pipeline for {input_source}")
        detector = PCBDetector(self.model_path, self.deploy_cfg)
        results, metadata = detector.detect(input_source, output_dir)
        logger.info("Deployment Pipeline Finished")
        return results, metadata

    def export_model(self, format="onnx"):
        from src.export import export_model

        return export_model(self.model_path, format)
