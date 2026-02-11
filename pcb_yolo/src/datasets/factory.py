from src.datasets.loader import PCBDataset
from src.datasets.augmentations import Augmentor
import torch
import random
import numpy as np


class DatasetFactory:
    @staticmethod
    def create_dataloader(img_dir, label_dir, batch_size=16, mode="train", seed=42):
        """
        Factory method to create a dataloader with specified augmentations.
        """
        augmentor = Augmentor(mode=mode)
        dataset = PCBDataset(img_dir, label_dir, transform=augmentor)

        # Set generator for reproducibility
        g = torch.Generator()
        g.manual_seed(seed)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            worker_init_fn=lambda worker_id: DatasetFactory._seed_worker(
                worker_id, seed
            ),
            generator=g,
        )

    @staticmethod
    def _seed_worker(worker_id, seed):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
