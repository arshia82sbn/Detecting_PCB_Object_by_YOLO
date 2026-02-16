import os
import pytest
import shutil
import subprocess
import json

# Set current directory to pcb_yolo for tests
@pytest.fixture(scope="module", autouse=True)
def setup_test_env():
    # Ensure we are in the right directory or have PYTHONPATH set
    orig_dir = os.getcwd()
    if os.path.basename(orig_dir) != 'pcb_yolo':
        if os.path.exists('pcb_yolo'):
            os.chdir('pcb_yolo')

    yield
    os.chdir(orig_dir)

def create_mock_data_for_test():
    cv2 = pytest.importorskip("cv2", reason="OpenCV runtime (libGL) not available", exc_type=ImportError)
    import numpy as np
    base = "data/mock"
    for split in ['train', 'val', 'test']:
        img_dir = f"{base}/{split}/images"
        lbl_dir = f"{base}/{split}/labels"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(2):
            img_path = f"{img_dir}/img_{i}.jpg"
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
            with open(f"{lbl_dir}/img_{i}.txt", 'w') as f:
                f.write("0 0.5 0.5 0.2 0.2\n")

    # Update mock_data.yaml to use absolute path
    abs_path = os.path.abspath(base)
    with open("configs/mock_data.yaml", 'w') as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("\nnc: 1\nnames: ['defect']\n")

def test_full_pipeline_smoke():
    """
    Test the full pipeline: Prepare -> Train -> Export -> Infer
    """
    pytest.importorskip("cv2", reason="OpenCV runtime (libGL) not available", exc_type=ImportError)
    create_mock_data_for_test()
    # 1. Prepare
    assert os.path.exists("configs/data_config.yaml")

    # 2. Train (1 epoch)
    train_cmd = [
        "python3", "-m", "src.train",
        "--config", "configs/test_train_config.yaml",
        "--data", "configs/mock_data.yaml",
        "--seed", "42",
        "--deterministic"
    ]
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # List experiments for debugging
    if os.path.exists("experiments"):
        print(f"Experiments contents: {os.listdir('experiments')}")
        for d in os.listdir('experiments'):
            if os.path.isdir(f"experiments/{d}"):
                print(f"experiments/{d} contents: {os.listdir(f'experiments/{d}')}")

    assert result.returncode == 0, f"Training failed: {result.stderr}"

    best_model = "experiments/test_train/weights/best.pt"
    # The trainer also registers it at experiments/test_train/model.pt
    registered_model = "experiments/test_train/model.pt"
    assert os.path.exists(registered_model)

    # 3. Export
    export_cmd = [
        "python3", "-m", "src.export",
        "--model", registered_model,
        "--format", "onnx"
    ]
    result = subprocess.run(export_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Export failed: {result.stderr}"
    assert os.path.exists("experiments/test_train/model.onnx")

    # 4. Inference
    infer_cmd = [
        "python3", "-m", "src.inference.detector",
        "--model", registered_model,
        "--input", "data/mock/test/images",
        "--output", "experiments/test_predictions",
        "--config", "configs/deploy_config.yaml"
    ]
    result = subprocess.run(infer_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Inference failed: {result.stderr}"

    assert os.path.exists("experiments/test_predictions/detections.json")
    with open("experiments/test_predictions/detections.json", 'r') as f:
        detections = json.load(f)
        assert isinstance(detections, list)
        assert len(detections) > 0
        assert "image" in detections[0]
        assert "detections" in detections[0]
