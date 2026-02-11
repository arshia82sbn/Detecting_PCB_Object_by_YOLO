from src.models.yolo_model import YOLOModel


class ModelFactory:
    @staticmethod
    def get_model(model_config):
        """
        Factory method to get a model based on configuration.
        """
        model_type = model_config.get("model_type", "yolov8s.pt")
        if "yolo" in model_type.lower():
            return YOLOModel(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
