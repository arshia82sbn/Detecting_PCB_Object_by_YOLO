import pytest
import numpy as np
from src.utils.metrics import calculate_iou, compute_ap, compute_f1

def test_calculate_iou():
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = calculate_iou(box1, box2)
    # intersection: [5, 5, 10, 10] -> area 25
    # union: 100 + 100 - 25 = 175
    expected_iou = 25 / 175
    assert pytest.approx(iou) == expected_iou

    box3 = [20, 20, 30, 30]
    assert calculate_iou(box1, box3) == 0

def test_compute_f1():
    p = 0.8
    r = 0.5
    f1 = compute_f1(p, r)
    expected_f1 = 2 * (0.8 * 0.5) / (0.8 + 0.5)
    assert pytest.approx(f1) == expected_f1

def test_compute_ap():
    # Simple case: precision constant at 1.0
    recall = np.linspace(0, 1, 10)
    precision = np.ones(10)
    ap = compute_ap(recall, precision)
    assert pytest.approx(ap, abs=0.1) == 1.0
