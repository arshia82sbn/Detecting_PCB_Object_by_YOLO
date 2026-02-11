import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) using the 11-point interpolation.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    x = np.linspace(0, 1, 11)
    # np.trapz was renamed to np.trapezoid in NumPy 2.0
    if hasattr(np, "trapezoid"):
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    else:
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap


def compute_f1(precision, recall):
    """
    Compute F1 score.
    """
    return 2 * (precision * recall) / (precision + recall + 1e-16)


def plot_pr_curve(precision, recall, ap, save_path):
    """
    Plot and save Precision-Recall curve for publication.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"mAP@0.5 = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
