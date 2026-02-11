import os
import cv2
import torch
from torch.utils.data import Dataset
from src.datasets.augmentations import Augmentor


class PCBDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = [
            f for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        label_path = os.path.join(
            self.label_dir, self.img_names[idx].rsplit(".", 1)[0] + ".txt"
        )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.split())
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(cls))

        if self.transform:
            augmented = self.transform(
                image=image, bboxes=bboxes, class_labels=class_labels
            )
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            class_labels = augmented["class_labels"]

        return image, bboxes, class_labels


class DatasetFactory:
    @staticmethod
    def create_dataloader(img_dir, label_dir, batch_size=16, mode="train"):
        augmentor = Augmentor(mode=mode)
        dataset = PCBDataset(img_dir, label_dir, transform=augmentor)
        # Note: YOLO usually expects a specific directory structure and handles loading itself.
        # This custom loader is for non-YOLO specific tasks or advanced customization.
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(mode == "train")
        )
