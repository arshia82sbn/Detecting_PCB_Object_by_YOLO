import albumentations as A
import cv2
import numpy as np
import random


class Augmentor:
    """
    Advanced augmentations using Albumentations.
    """

    def __init__(self, mode="train"):
        self.mode = mode
        if mode == "train":
            self.transform = A.Compose(
                [
                    A.RandomRotate90(),
                    A.Flip(),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                    ),
                    A.RandomBrightnessContrast(p=0.2),
                    A.HueSaturationValue(p=0.2),
                    A.Blur(blur_limit=3, p=0.1),
                    A.GaussNoise(p=0.1),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )
        else:
            self.transform = A.Compose(
                [
                    # Basic validation transforms
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

    def __call__(self, image, bboxes, class_labels):
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
