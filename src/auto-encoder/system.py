import torch
from torch.nn import MSELoss
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from .dataset import AutoEncoderDataset
from .model import Encoder, Decoder


class AutoEncoderSystem(pl.LightningModule):
    def __init__(self):
        super(AutoEncoderSystem, self).__init__()
        self.encoder = Encoder().double()
        self.decoder = Decoder().double()
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
    def __init__(self, trn_data_path, val_data_path, batch_size: int = 128):
        super().__init__()
        self.val_dataset = None
        self.trn_dataset = None

        trn_samples = list(trn_data_path.glob('*.npy'))
        val_samples = list(val_data_path.glob('*.npy'))

        trn_processed_samples = []
        val_processed_samples = []

        for i, sample in tqdm(trn_samples):
            trn_processed_samples.append(np.load(sample))

        for sample in tqdm(val_samples):
            val_processed_samples.append(np.load(sample))

        self.train_data = np.concatenate(trn_processed_samples)
        self.val_data = np.concatenate(val_processed_samples)

        min_train = self.train_data.min(axis=0)
        max_train = self.train_data.max(axis=0)

        self.min_train = min_train
        self.max_train = max_train

        # min-max normalization
        self.train_data = (self.train_data - min_train) / (max_train - min_train)
        self.val_data = (self.val_data - min_train) / (max_train - min_train)

        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        self.trn_dataset = AutoEncoderDataset(self.train_data)
        self.val_dataset = AutoEncoderDataset(self.val_data)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
