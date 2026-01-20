import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def make_noisy_labels(y, num_classes, noise_ratio, mode="sym", seed=0):
    rng = np.random.RandomState(seed)
    y = np.asarray(y).copy()
    n = len(y)
    n_noisy = int(noise_ratio * n)
    idx = rng.permutation(n)[:n_noisy]

    if mode == "sym":
        for i in idx:
            old = y[i]
            new = rng.randint(num_classes - 1)
            if new >= old:
                new += 1
            y[i] = new
    else:
        # CIFAR-10 asymmetric noise mapping (standard)
        # truck->automobile, bird->airplane, cat<->dog, deer->horse
        if num_classes != 10:
            raise ValueError("asym noise is typically defined for CIFAR-10 only.")
        mapping = {9:1, 2:0, 3:5, 5:3, 4:7}  # class ids: truck=9, auto=1, bird=2, plane=0, cat=3, dog=5, deer=4, horse=7
        for i in idx:
            y[i] = mapping.get(int(y[i]), int(y[i]))
    return y

class CIFARNoisy(Dataset):
    def __init__(self, root, train, cifar100=False, transform=None, noise_ratio=0.5, noise_mode="sym", seed=0):
        self.transform = transform
        base = datasets.CIFAR100 if cifar100 else datasets.CIFAR10
        self.ds = base(root=root, train=train, download=True)
        self.data = self.ds.data
        self.targets_clean = np.asarray(self.ds.targets, dtype=np.int64)

        self.num_classes = 100 if cifar100 else 10
        if train:
            self.targets = make_noisy_labels(self.targets_clean, self.num_classes, noise_ratio, noise_mode, seed)
        else:
            self.targets = self.targets_clean.copy()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = int(self.targets[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x, y, idx
