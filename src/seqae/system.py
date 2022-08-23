from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from .model import Encoder, Decoder
from .dataset import SAEDataset


class SequentialAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: 40):
        super(SequentialAutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_size=hidden_size)
        self.decoder = Decoder(output_dim=output_dim, hidden_size=hidden_size)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_nb):
        x = batch
        x_hat = self.forward(x)
        loss = self.loss(x_hat, x)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x = batch
        x_hat = self.forward(x)
        loss = self.loss(x_hat, x)
        self.log('train/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class SAEDataModule(pl.LightningDataModule):
    def __init__(self, trn_folder: str, val_folder: str, stride: int = 5, window_size: int = 10, batch_size: int = 32):
        super().__init__()
        self.trn_folder = trn_folder
        self.val_folder = val_folder
        self.batch_size = batch_size
        self.stride = stride
        self.window_size = window_size
        self.trn_dataset = None  # type: FeedforwardDataset
        self.val_dataset = None  # type: FeedforwardDataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.trn_dataset = SAEDataset(Path(self.trn_folder).glob('*.npy'), window_size=self.window_size,
                                      stride=self.stride)
        self.val_dataset = SAEDataset(Path(self.val_folder).glob('*.npy'), window_size=self.window_size,
                                      stride=self.stride)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)