from ultralytics import YOLO
import os

class YOLOModel:
    """
    Wrapper around Ultralytics YOLO model.
    """
    def __init__(self, model_config):
        self.config = model_config
        self.model = YOLO(self.config['model_type'])

    def train(self, data_config_path, train_config):
        # Remove 'data' from train_config if it's there to avoid multiple values
        conf = train_config.copy()
        if 'data' in conf:
            del conf['data']

        results = self.model.train(
            data=data_config_path,
            **conf
        )
        return results

    def predict(self, source, **kwargs):
        return self.model.predict(source, **kwargs)

    def export(self, format='onnx', **kwargs):
        return self.model.export(format=format, **kwargs)

    def load_weights(self, weights_path):
        self.model = YOLO(weights_path)
