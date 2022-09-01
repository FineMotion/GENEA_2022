from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.nn import MSELoss, L1Loss
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import Wav2GestDataset
from .model import GRUEncoder, GRUAttentionDecoder
from src.base import BaseDataModule


class Wav2GestSystem(pl.LightningModule):
    def __init__(self, input_dim: int = 26, output_dim: int = 164):
        super(Wav2GestSystem, self).__init__()
        self.encoder = GRUEncoder(in_features=input_dim, hidden_size=150)
        self.decoder = GRUAttentionDecoder(enc_hidden=150, out_features=output_dim, hidden_dim=150)
        self.mse = MSELoss()
        self.l1 = L1Loss()
        self.gamma = 0.6

    def forward(self, x, p):
        output, hidden = self.encoder(x)
        predicted_poses = self.decoder(output, hidden, p)
        return predicted_poses

    def custom_loss(self, output, target, history, stage='train'):

        mse_loss = self.mse(output, target)
        self.log(f'{stage}/mse_loss', mse_loss)

        # output_velocity = output[:, 1:, :] - output[:, :-1, :]
        # target_velocity = target[:, 1:, :] - target[:, :-1, :]
        # velocity_loss = self.mse(output_velocity, target_velocity)
        # self.log(f'{stage}/velocity_loss', velocity_loss)

        # loss = mse_loss + 0.6 * velocity_loss
        loss = mse_loss
        self.log(f'{stage}/loss', loss)

        return loss

    def training_step(self, batch, batch_nb):
        x, y, p = batch

        # before autoregression
        # if self.current_epoch < 20:
        #     pred_poses = self.forward(x, p.new_zeros(p.shape))
        # else:
        pred_poses = self.forward(x, p)

        loss = self.custom_loss(pred_poses, y, p, 'train')
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y, p = batch
        pred_poses = self.forward(x, p)
        loss = self.custom_loss(pred_poses, y, p, 'val')
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class Wav2GestDataModule(BaseDataModule):
    def __init__(self, trn_folder: str, val_folder: str, train_metadata_path: str = None, val_metadata_path: str = None,
                 filter_user: int = None, only_fingers: bool = False, stride: int = 15, batch_size: int = 32,
                 audio_framerate: float = 30.):
        super().__init__(trn_folder, val_folder, train_metadata_path, val_metadata_path, filter_user, only_fingers)
        self.batch_size = batch_size
        self.stride = stride
        self.audio_framerate = audio_framerate
        self.trn_dataset = None  # type: Wav2GestDataset
        self.val_dataset = None  # type: Wav2GestDataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.trn_dataset = Wav2GestDataset(data_files=self.train_samples, stride=self.stride,
                                           wav_fps=self.audio_framerate)
        self.val_dataset = Wav2GestDataset(data_files=self.val_samples, stride=self.stride,
                                           wav_fps=self.audio_framerate)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)
