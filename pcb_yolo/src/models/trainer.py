import os
import random
import numpy as np
import torch
import shutil
from src.utils.logger import logger
from src.models.factory import ModelFactory

class PCBYOLOTrainer:
    """
    Trainer class for fine-tuning YOLO models on PCB datasets.
    """
    def __init__(self, model_config, train_config, data_config_path):
        self.model_config = model_config
        self.train_config = train_config
        self.data_config_path = data_config_path

        # Set seeds for reproducibility
        self._set_seeds(self.train_config.get('seed', 42))

        self.model = ModelFactory.get_model(model_config)

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic behavior in cuDNN
        if self.train_config.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed}")

    def run_training(self):
        """
        Run the training process with the given configuration.
        """
        logger.info("Starting training pipeline...")

        # Prepare training arguments
        train_args = self.train_config.copy()
        train_args['data'] = self.data_config_path
        train_args['imgsz'] = self.model_config.get('imgsz', 640)

        # Ensure project path is absolute to avoid site-packages issues
        project = train_args.get('project', 'runs/detect')
        train_args['project'] = os.path.abspath(project)

        # Default mixed precision if not set
        if 'amp' not in train_args:
            train_args['amp'] = True

        try:
            # Ultralytics handles its own training loop
            results = self.model.train(
                data_config_path=self.data_config_path,
                train_config=train_args
            )
            logger.info("Training completed successfully.")

            # Identify the best model path
            # Ultralytics saves to {project}/{name}/weights/best.pt
            project = train_args.get('project', 'runs/detect')
            name = train_args.get('name', 'train')

            # If relative, make it absolute to be sure
            # Use the absolute path of the current directory to avoid saving to site-packages
            abs_project = os.path.abspath(project)

            run_dir = os.path.join(abs_project, name)
            best_model_path = os.path.join(run_dir, 'weights', 'best.pt')

            if os.path.exists(best_model_path):
                # Register final model to a fixed location in the run dir
                standard_model_path = os.path.join(run_dir, 'model.pt')
                shutil.copy(best_model_path, standard_model_path)
                logger.info(f"Best model registered at {standard_model_path}")
            else:
                logger.warning(f"Could not find best model at {best_model_path}")

            return results
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e
