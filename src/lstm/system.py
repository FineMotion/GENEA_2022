from typing import Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from src.lstm.model import LstmModel
from src.lstm.dataset import LstmDataset
from src.base import BaseDataModule


class LstmSystem(pl.LightningModule):
    def __init__(self, in_features: int = 26, out_features: int =164):
        super().__init__()
        self.model = LstmModel(in_features=in_features, out_features=out_features, hidden_size=50)
        self.mse = nn.MSELoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.mse(pred, y)
        self.log('train/loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.mse(pred, y)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class LstmDataModule(BaseDataModule):
    def __init__(self, trn_folder: str, val_folder: str, train_metadata_path: str, val_metadata_path: str,
                 filter_user: int = None, stride: int = 15, window_size: int = 30, batch_size: int = 32):
        super().__init__(trn_folder, val_folder, train_metadata_path, val_metadata_path, filter_user)
        self.batch_size = batch_size
        self.stride = stride
        self.window_size = window_size
        self.trn_dataset = None  # type: FeedforwardDataset
        self.val_dataset = None  # type: FeedforwardDataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.trn_dataset = LstmDataset(self.train_samples,
                                       stride=self.stride, window_size=self.window_size)
        self.val_dataset = LstmDataset(self.val_samples,
                                       stride=self.stride, window_size=self.window_size)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)