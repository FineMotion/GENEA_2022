from pathlib import Path
from typing import Iterator, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from math import floor


class ReCellSeqDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], seq_len: int = 30, stride: int = 5, win_size: int = 61,
                 wav_fps: float = 30., gest_fps: float = 30.):
        self.X = []
        self.Y = []
        self.storage = []
        self.seq_len = seq_len
        self.stride = stride
        self.win_size = win_size
        self.wav_fps = wav_fps
        self.gest_fps = gest_fps

        for i, data_file in tqdm(enumerate(data_files)):
            self.process_file(data_file, i)

    def process_file(self, file_path: Union[str, Path], file_idx: int):
        data = np.load(file_path)
        x = data['X']
        y = data['Y']

        paddings = np.zeros((self.win_size // 2, x.shape[1]), dtype=float)
        padded_x = np.concatenate([paddings, x, paddings])

        for start in range(0, y.shape[0], self.stride):
            self.storage.append((file_idx, start))

        self.X.append(padded_x)
        self.Y.append(y)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, item):
        file_idx, start_gest = self.storage[item]
        file_x, file_y = self.X[file_idx], self.Y[file_idx]

        end_gest = min(start_gest + self.seq_len, file_y.shape[0])
        # start_wav = floor(start_gest * self.wav_fps / self.gest_fps)

        x = np.zeros((self.seq_len, self.win_size, file_x.shape[1]), dtype=float)
        y = np.zeros((self.seq_len, file_y.shape[1]), dtype=float)
        h = file_y[start_gest - 1] if start_gest > 0 else np.zeros((file_y.shape[1]), dtype=float)

        for i in range(start_gest, end_gest):
            y[i - start_gest] = file_y[i]
            start_wav = floor(i * self.wav_fps / self.gest_fps)
            end_wav = min(start_wav + self.win_size, file_x.shape[0])
            x[i - start_gest, :end_wav - start_wav] = file_x[start_wav:end_wav]

        # x - seq_len, win_size, in_dim
        # y - seq_len, out_dim
        # h - out_dim
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(h)

    @staticmethod
    def collate_fn(batch):
        x, y, h = list(zip(*batch))
        # x - seq_len, batch_size, win_size, in_dim
        # y - seq_len, batch_size, out_dim
        # h - batch_size, out_dim
        return torch.stack(x, dim=1), torch.stack(y, dim=1), torch.stack(h, dim=0)