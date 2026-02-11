import logging
import os
import sys
from datetime import datetime


class Logger:
    """
    Singleton Logger for structured logging.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        self.logger = logging.getLogger("pcb_yolo")
        self.logger.setLevel(logging.INFO)

        # Create experiments/logs directory if it doesn't exist
        log_dir = "experiments/logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(
            log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Clear existing handlers if any (to avoid duplicate logs)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger


# Global logger instance
logger = Logger().get_logger()
