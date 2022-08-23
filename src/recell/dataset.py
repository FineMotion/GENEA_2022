from pathlib import Path
from typing import Iterator, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class ReCellDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str,Path]], window_size: int = 61):
        self.X = []
        self.H = []
        self.Y = []
        self.window_size = window_size

        for data_file in data_files:
            self.process_file(data_file)

    def process_file(self, file_path: Union[str, Path]):
        data = np.load(file_path)
        x, y = data['X'], data['Y']
        paddings = np.zeros((self.window_size // 2, x.shape[1]), dtype=float)
        padded_x = np.concatenate([paddings,x,paddings])
        for i in range(y.shape[0]):
            sample_x = padded_x[i: i + self.window_size]
            sample_y = y[i]
            sample_h = y[i-1] if i > 0 else np.zeros(y.shape[1])

            self.X.append(sample_x)
            self.Y.append(sample_y)
            self.H.append(sample_h)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        x, y, h = self.X[item], self.Y[item], self.H[item]
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(h)

    @staticmethod
    def collate_fn(batch):
        x, y, h = list(zip(*batch))
        return torch.stack(x, dim=0), torch.stack(y, dim=0), torch.stack(h, dim=0)








