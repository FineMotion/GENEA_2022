import logging

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Iterator, Union
import numpy as np
from tqdm import tqdm


class LstmDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], stride: int = 1, window_size: int = 30):
        self.X = []
        self.Y = []
        self.window_size = window_size
        self.stride = stride
        self.storage = []
        logging.info('Loading dataset...')
        for file_idx, data_file in tqdm(enumerate(data_files)):
            self.process_file(data_file, file_idx)

    def process_file(self, file_path: Union[str, Path], file_idx: int):
        data = np.load(file_path)
        x, y = data['X'], data['Y']
        for start in range(0, y.shape[0], self.stride):
            self.storage.append((file_idx, start))
        self.X.append(x)
        self.Y.append(y)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, item):
        file_idx, start = self.storage[item]
        filex, filey = self.X[file_idx], self.Y[file_idx]
        end = min(len(filey), start + self.window_size)

        x = np.zeros((self.window_size, filex.shape[1]), dtype=float)
        y = np.zeros((self.window_size, filey.shape[1]), dtype=float)
        x[:end - start] = filex[start: end]
        y[:end - start] = filey[start: end]
        return torch.FloatTensor(x), torch.FloatTensor(y)

    @staticmethod
    def collate_fn(batch):
        x, y = list(zip(*batch))
        return torch.stack(x, dim=0), torch.stack(y, dim=0)
