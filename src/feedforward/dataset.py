import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Iterator, Union
import numpy as np


class FeedforwardDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], window_size: int = 61):
        self.X = []
        self.Y = []
        self.window_size = window_size
        for data_file in data_files:
            self.process_file(data_file)

    def process_file(self, file_path: Union[str, Path]):
        data = np.load(file_path)
        x, y = data['X'], data['Y']
        paddings = np.zeros((self.window_size // 2, x.shape[1]), dtype=float)
        padded_x = np.concatenate([paddings, x, paddings])
        for i in range(y.shape[0]):
            sample_x = padded_x[i: i + self.window_size]
            sample_y = y[i]
            self.X.append(sample_x)
            self.Y.append(sample_y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        x, y = self.X[item], self.Y[item]
        return torch.FloatTensor(x), torch.FloatTensor(y)

    @staticmethod
    def collate_fn(batch):
        x, y = list(zip(*batch))
        return torch.stack(x, dim=0), torch.stack(y, dim=0)
