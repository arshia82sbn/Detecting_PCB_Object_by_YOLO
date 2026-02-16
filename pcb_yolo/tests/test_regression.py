from src.models.yolo_model import YOLOModel


class _DummyYOLO:
    def __init__(self):
        self.train_call = None

    def train(self, **kwargs):
        self.train_call = kwargs
        return {"ok": True}


def test_yolo_model_train_data_override(monkeypatch):
    """
    Regression test: Ensure 'data' in train_config doesn't cause TypeError.
    """
    monkeypatch.setattr(YOLOModel, "_load_yolo", staticmethod(lambda _: _DummyYOLO()))

    model_cfg = {"model_type": "yolov8n.pt"}
    model = YOLOModel(model_cfg)

    train_cfg = {"epochs": 1, "data": "should_be_removed"}
    result = model.train("dataset.yaml", train_cfg)

    assert result == {"ok": True}
    assert model.model.train_call == {"data": "dataset.yaml", "epochs": 1}


def test_load_weights_replaces_model(monkeypatch):
    """
    Regression test: Ensure `load_weights` swaps model instance.
    """

    created = []

    def _factory(weight_name):
        obj = {"weights": weight_name}
        created.append(obj)
        return obj

    monkeypatch.setattr(YOLOModel, "_load_yolo", staticmethod(_factory))

    model = YOLOModel({"model_type": "first.pt"})
    model.load_weights("second.pt")

    assert created[0]["weights"] == "first.pt"
    assert model.model["weights"] == "second.pt"
