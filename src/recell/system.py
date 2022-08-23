from typing import Optional

import pytorch_lightning as pl
import torch.distributed.optim
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from .model import ReCellModel
from .dataset import ReCellDataset
from torch.nn import MSELoss
from src.base.system import BaseDataModule


class ReCellSystem(pl.LightningModule):
    def __init__(self, input_dim: int = 26, output_dim: int = 164, window_size: int = 61):
        super(ReCellSystem, self).__init__()
        self.model = ReCellModel(in_features=input_dim, out_features=output_dim, window_size=window_size)
        self.loss = MSELoss()

    def forward(self, x, h):
        pred = self.model(x, h)
        return pred

    def training_step(self, batch, batch_idx):
        x, y, h = batch
        if self.current_epoch < 10:
            h = h.new_zeros(h.shape)
        pred = self.forward(x, h)
        loss = self.loss(pred, y)
        self.log('train/loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, h = batch
        pred = self.forward(x, h)
        loss = self.loss(pred, y)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class ReCellDataModule(BaseDataModule):
    def __init__(self, trn_folder: str, val_folder: str, train_metadata_path: str, val_metadata_path: str,
                 filter_user: int = None, only_fingers: bool = False, batch_size: int = 32):
        super().__init__(trn_folder, val_folder, train_metadata_path, val_metadata_path, filter_user, only_fingers)
        self.val_folder = val_folder
        self.batch_size = batch_size
        self.trn_dataset = None  # type: ReCellDataset
        self.val_dataset = None  # type: ReCellDataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.trn_dataset = ReCellDataset(self.train_samples)
        self.val_dataset = ReCellDataset(self.val_samples)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)
