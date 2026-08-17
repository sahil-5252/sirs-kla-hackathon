import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    Loads paired (NoisyLR, GT) .npy files based on a split CSV.

    - NoisyLR: (128, 128) float32, values may exceed [0, 1] (NOT clipped/normalized)
    - GT:      (256, 256) float32, values in [0, 1]
    - Pairs are matched by identical filename.

    Geometric augmentations (applied identically to BOTH members of a pair,
    only for the 'train' split):
      - Random rotation by a multiple of 90 degrees (90/180/270)
      - Random horizontal flip (mirror)
      - Random vertical flip (mirror)
    """
    def __init__(self, csv_path, gt_dir, noisy_dir, split='train', augment=True):
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.augment = augment and (split == 'train')

        self.filenames = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.filenames.append(row['filename'])
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        gt = np.load(os.path.join(self.gt_dir, filename)).astype(np.float32)
        noisy = np.load(os.path.join(self.noisy_dir, filename)).astype(np.float32)

        if self.augment:
            # Same random choices for both members of the pair.
            k = random.choice([0, 1, 2, 3])               # 0/90/180/270 deg rotation
            if k:
                gt = np.rot90(gt, k)
                noisy = np.rot90(noisy, k)
            if random.random() > 0.5:                      # horizontal mirror
                gt = np.flip(gt, axis=1)
                noisy = np.flip(noisy, axis=1)
            if random.random() > 0.5:                      # vertical mirror
                gt = np.flip(gt, axis=0)
                noisy = np.flip(noisy, axis=0)
            gt = np.ascontiguousarray(gt)
            noisy = np.ascontiguousarray(noisy)

        noisy = noisy[np.newaxis, ...]  # (1, 128, 128)
        gt = gt[np.newaxis, ...]        # (1, 256, 256)
        return torch.from_numpy(noisy), torch.from_numpy(gt)


class TestDataset(Dataset):
    """
    Loads NoisyLR .npy files for inference. No GT available.
    """
    def __init__(self, noisy_dir):
        self.noisy_dir = noisy_dir
        self.filenames = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        noisy = np.load(os.path.join(self.noisy_dir, filename)).astype(np.float32)
        noisy = noisy[np.newaxis, ...]
        return torch.from_numpy(noisy), filename