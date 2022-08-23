import logging

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Iterator, Union
import numpy as np
from tqdm import tqdm


class SAEDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], window_size: int, stride: int):
        self.X = []
        self.window_size = window_size
        self.stride = stride
        self.storage = []
        logging.info('Loading dataset...')
        for file_idx, data_file in tqdm(enumerate(data_files)):
            self.process_file(data_file, file_idx)

    def process_file(self, file_path: Union[str, Path], file_idx: int):
        data = np.load(file_path)
        for start in range(0, data.shape[0], self.stride):
            self.storage.append((file_idx, start))
        self.X.append(data)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, item):
        file_idx, start = self.storage[item]
        data = self.X[file_idx]
        end = min(len(data), start + self.window_size)

        x = np.zeros((self.window_size, data.shape[1]), dtype=float)
        x[:end - start] = data[start: end]
        return torch.FloatTensor(x)

    # @staticmethod
    # def collate_fn(batch):
    #     x = list(zip(*batch))
    #     return torch.stack(x, dim=0)
