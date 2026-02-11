from src.utils.helpers import load_yaml
from src.models.trainer import PCBYOLOTrainer
from src.utils.logger import logger
import os

class TrainPipeline:
    """
    Orchestrates the training process.
    """
    def __init__(self, data_cfg, model_cfg, train_cfg):
        self.data_cfg_path = data_cfg
        self.model_cfg = load_yaml(model_cfg)
        self.train_cfg = load_yaml(train_cfg)

    def run(self):
        logger.info("Initializing Training Pipeline")
        trainer = PCBYOLOTrainer(
            model_config=self.model_cfg,
            train_config=self.train_cfg,
            data_config_path=self.data_cfg_path
        )
        results = trainer.run_training()
        logger.info("Training Pipeline Finished")
        return results
