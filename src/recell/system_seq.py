from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from src.base import BaseDataModule
from src.recell.model import ReCellModel
from src.recell.dataset_seq import ReCellSeqDataset
import random


class ReCellSeqSystem(pl.LightningModule):
    def __init__(self, input_dim: int = 26, output_dim: int = 164, window_size: int = 61):
        super(ReCellSeqSystem, self).__init__()
        self.model = ReCellModel(in_features=input_dim, out_features=output_dim, window_size=window_size)
        self.mse = MSELoss()

    def forward(self, x, h):
        pred = self.model(x, h)
        return pred

    def training_step(self, batch, batch_idx):
        x, y, h = batch
        # x - seq_len, batch_size, win_size, in_dim
        # y - seq_len, batch_size, out_dim
        # h - batch_size, out_dim
        if self.current_epoch < 10:
            h = h.new_zeros(h.shape)

        preds = []
        for i in range(y.shape[0]):
            pred = self.model(x[i], h)
            preds.append(pred)
            tf = random.random()
            if tf > 0.8:
                if self.current_epoch < 10:
                    h = h.new_zeros(h.shape)
                else:
                    h = y[i]
            else:
                h = pred
            # here we can make h = 0, h = pred, h = y[i]
            # if self.current_epoch < 10:
            #     h = h.new_zeros(h.shape)
            # else:
            #     h = pred

        preds = torch.stack(preds, dim=0)
        loss = self.custom_loss(preds, y, 'train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, h = batch
        # x - seq_len, batch_size, win_size, in_dim
        # y - seq_len, batch_size, out_dim
        # h - batch_size, out_dim

        preds = []
        for i in range(y.shape[0]):
            pred = self.model(x[i], h)
            preds.append(pred)
            h = pred

        preds = torch.stack(preds, dim=0)
        loss = self.custom_loss(preds, y, 'val')
        return {'loss': loss}

    def custom_loss(self, output, target, stage='train'):
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)

        mse_loss = self.mse(output, target)
        self.log(f'{stage}/mse_loss', mse_loss)

        output_velocity = output[:, 1:, :] - output[:, :-1, :]
        target_velocity = target[:, 1:, :] - target[:, :-1, :]
        velocity_loss = self.mse(output_velocity, target_velocity)
        self.log(f'{stage}/velocity_loss', velocity_loss)

        loss = mse_loss + 0.6 * velocity_loss

        self.log(f'{stage}/loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class ReCellSeqDataModule(BaseDataModule):
    def __init__(self, trn_folder: str, val_folder: str, train_metadata_path: str, val_metadata_path: str,
                 filter_user: int = None, only_fingers: bool = False, stride: int = 15, batch_size: int = 32,
                 audio_framerate: float = 30., window_size: int = 10, seq_len: int = 125):
        super().__init__(trn_folder, val_folder, train_metadata_path, val_metadata_path, filter_user, only_fingers)
        self.batch_size = batch_size
        self.stride = stride
        self.audio_framerate = audio_framerate
        self.trn_dataset = None  # type: ReCellSeqDataset
        self.val_dataset = None  # type: ReCellSeqDataset
        self.window_size = window_size
        self.seq_len = seq_len

    def setup(self, stage: Optional[str] = None) -> None:
        self.trn_dataset = ReCellSeqDataset(data_files=self.train_samples, stride=self.stride,
                                            wav_fps=self.audio_framerate, seq_len=self.seq_len, win_size=self.window_size)
        self.val_dataset = ReCellSeqDataset(data_files=self.val_samples, stride=self.stride,
                                            wav_fps=self.audio_framerate, seq_len=self.seq_len, win_size=self.window_size)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)