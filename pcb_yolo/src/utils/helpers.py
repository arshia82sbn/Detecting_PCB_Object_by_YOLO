import yaml
import json
import os
import cv2
import matplotlib.pyplot as plt

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(data, f)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def draw_boxes(image, boxes, labels, scores=None):
    # Simple visualization helper
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = labels[i]
        if scores is not None:
            label += f" {scores[i]:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    return image

def save_image(image, filepath):
    cv2.imwrite(filepath, image)
