import logging
import math

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Iterator, Union
import numpy as np
from tqdm import tqdm


class Seq2seqDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], stride: int=15):
        self.X = []
        self.Y = []
        self.window_size = 30
        self.stride = stride
        self.history = 15
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
        end = min(len(filey), start+self.window_size)
        past = max(0, start - self.history)

        x = np.zeros((self.window_size, filex.shape[1]), dtype=float)
        y = np.zeros((self.window_size, filey.shape[1]), dtype=float)
        h = np.zeros((self.history, filey.shape[1]), dtype=float)

        x[:end - start] = filex[start: end]
        y[:end - start] = filey[start: end]
        if past != start:
            h[past - start:] = filey[past:start]

        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(h)

    @staticmethod
    def collate_fn(batch):
        x, y, h = list(zip(*batch))
        return torch.stack(x, dim=1), torch.stack(y, dim=1), torch.stack(h, dim=1)


class ExpandedSeq2SeqDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], seq_len: int, stride: int, history: int,
                 audio_framerate: float = 30., motion_framerate: float = 30.):
        self.X = []
        self.Y = []
        self.storage = []
        self.stride = stride
        self.motion_seq_len = seq_len
        self.audio_seq_len = math.floor((seq_len + 2*history) * audio_framerate / motion_framerate)
        self.history = history
        self.audio_framerate = audio_framerate
        self.motion_framerate = motion_framerate
        for i, data_file in tqdm(enumerate(data_files)):
            self.process_file(data_file, i)

    def process_file(self, file_path:str, file_idx: int):
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
        file_idx, start = self.storage[item]
        file_audio, file_motion = self.X[file_idx], self.Y[file_idx]
        end = min(len(file_motion), start + self.motion_seq_len)
        past = max(0, start - self.history)
        future = min(len(file_motion), end + self.history)

        audio_start = math.floor(past * self.audio_framerate / self.motion_framerate)
        audio_end = math.floor(future * self.audio_framerate / self.motion_framerate)

        x = np.zeros((self.audio_seq_len, file_audio.shape[1]), dtype=float)
        y = np.zeros((self.motion_seq_len, file_motion.shape[1]), dtype=float)
        h = np.zeros((self.history, file_motion.shape[1]), dtype=float)

        x[:audio_end - audio_start] = file_audio[audio_start:audio_end]
        y[:end-start] = file_motion[start:end]
        if past != start:
            h[past - start:] = file_motion[past:start]

        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(h)

    @staticmethod
    def collate_fn(batch):
        x, y, h = list(zip(*batch))
        return torch.stack(x, dim=1), torch.stack(y, dim=1), torch.stack(h, dim=1)
