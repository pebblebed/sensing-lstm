import numpy as np
import torch

def white_balance_and_squash(x: np.ndarray):
    assert len(x.shape) == 2
    (rows, cols) = x.shape
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    return ((x - means) / stds), means, stds

def unbalance_and_unsquash(x: np.ndarray, means: np.ndarray, stds: np.ndarray):
    return (x * stds) + means

# E.g., bal, means, stds = white_balance_and_squash(raw)

from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, npdata: np.ndarray, stride: int, tail: int):
        super().__init__()
        npbal, means, stds = white_balance_and_squash(npdata)
        self.data = torch.from_numpy(npbal).float()
        self.means = torch.from_numpy(means).float()
        self.stds = torch.from_numpy(stds).float()
        self.stride = stride
        self.tail = tail

    def __len__(self):
        return int(len(self.data) / (self.stride + self.tail))

    def __getitem__(self, idx: int):
        # x: first _stride_ entries
        x_start = idx * (self.stride + self.tail)
        y_start = x_start + self.tail
        x_end = x_start + self.stride
        y_end = y_start + self.stride

        x = self.data[x_start:x_end]
        y = self.data[y_start:y_end]
        assert x.shape == y.shape
        return x, y