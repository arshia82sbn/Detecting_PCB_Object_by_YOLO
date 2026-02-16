import os


class YOLOModel:
    """
    Wrapper around Ultralytics YOLO model.
    """

    def __init__(self, model_config):
        self.config = model_config
        self.model = self._load_yolo(self.config["model_type"])

    @staticmethod
    def _load_yolo(weights_or_model_type):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - env-specific dependency issue
            raise ImportError(
                "Ultralytics YOLO dependencies are unavailable. "
                "Install ultralytics and system OpenCV libs (e.g., libGL)."
            ) from exc

        return YOLO(weights_or_model_type)

    def train(self, data_config_path, train_config):
        # Remove 'data' from train_config if it's there to avoid multiple values
        conf = train_config.copy()
        if "data" in conf:
            del conf["data"]

        results = self.model.train(data=data_config_path, **conf)
        return results

    def predict(self, source, **kwargs):
        return self.model.predict(source, **kwargs)

    def export(self, format="onnx", **kwargs):
        return self.model.export(format=format, **kwargs)

    def load_weights(self, weights_path):
        self.model = self._load_yolo(weights_path)
