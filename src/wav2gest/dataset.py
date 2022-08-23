from pathlib import Path
from typing import Iterator, Union

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from math import floor
import torch


class Wav2GestDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], stride: int,
                 present_len: int = 30, past_len: int = 15, future_len: int = 15,
                 wav_fps: float = 30., gest_fps: float = 30.):
        self.X = []
        self.Y = []
        self.storage = []
        self.present_len = present_len
        self.past_len = past_len
        self.future_len = future_len
        self.wav_len = floor((present_len + past_len + future_len) * wav_fps / gest_fps)
        self.wav_fps = wav_fps
        self.gest_fps = gest_fps
        self.stride = stride
        for i, data_file in tqdm(enumerate(data_files)):
            self.process_file(data_file, i)

    def process_file(self, file_path: Union[str, Path], file_idx: int):
        data = np.load(file_path)
        x = data['X']
        y = data['Y']

        for start in range(0, y.shape[0], self.stride):
            self.storage.append((file_idx, start))

        self.X.append(x)
        self.Y.append(y)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, item):
        file_idx, start_gest = self.storage[item]
        file_x, file_y = self.X[file_idx], self.Y[file_idx]

        end_gest = min(len(file_y), start_gest + self.present_len)
        past_gest = max(start_gest - self.past_len, 0)
        future_gest = min(len(file_y), end_gest + self.future_len)

        start_wav = floor(past_gest * self.wav_fps / self.gest_fps)
        end_wav = floor(future_gest * self.wav_fps / self.gest_fps)
        if end_wav - start_wav > self.wav_len:
            end_wav -= (end_wav - start_wav - self.wav_len)

        x = np.zeros((self.wav_len, file_x.shape[1]), dtype=float)
        y = np.zeros((self.present_len, file_y.shape[1]), dtype=float)
        h = np.zeros((self.past_len, file_y.shape[1]), dtype=float)

        x[:end_wav - start_wav] = file_x[start_wav:end_wav]
        y[:end_gest - start_gest] = file_y[start_gest: end_gest]
        if past_gest != start_gest:
            h[past_gest - start_gest:] = file_y[past_gest:start_gest]

        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(h)

    @staticmethod
    def collate_fn(batch):
        x, y, h = list(zip(*batch))
        return torch.stack(x, dim=0), torch.stack(y, dim=0), torch.stack(h, dim=0)