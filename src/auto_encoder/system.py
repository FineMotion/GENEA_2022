import torch
from torch.nn import MSELoss
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import os

from .utils import *
from .dataset import AutoEncoderDataset
from .model import Encoder, Decoder


class AutoEncoderSystem(pl.LightningModule):
    def __init__(self, input_dim, frames_count, output_dim, hidden_dim):
        super(AutoEncoderSystem, self).__init__()
        self.encoder = Encoder(input_dim, frames_count, output_dim, hidden_dim).double()
        self.decoder = Decoder(input_dim, frames_count, output_dim, hidden_dim).double()
        self.loss = MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_poses = self.forward(x)
        loss = self.loss(pred_poses, y)
        self.log('train/loss', loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_poses = self.forward(x)
        loss = self.loss(pred_poses, y)
        self.log('val/loss', loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self, trn_data_path, val_data_path, serialize_dir, batch_size: int = 128, norm_mode="min"):
        super().__init__()
        self.val_dataset = None
        self.trn_dataset = None

        assert norm_mode in ["min", "mean"]

        self.train_data = get_data_by_file(trn_data_path)
        self.val_data = get_data_by_file(val_data_path)

        min_train = self.train_data.min(axis=0) if norm_mode == "min" else self.train_data.mean(axis=0)
        max_train = self.train_data.max(axis=0)

        self.min_train = min_train
        self.max_train = max_train

        with open(os.path.join(serialize_dir, 'min_max.txt'), 'w') as f:
            f.write(str(self.min_train) + " " + str(self.max_train))

        # min-max (mean-max) normalization
        self.train_data = normalize(self.train_data, min_train, max_train)
        self.val_data = normalize(self.val_data, min_train, max_train)

        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        self.trn_dataset = AutoEncoderDataset(self.train_data)
        self.val_dataset = AutoEncoderDataset(self.val_data)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
