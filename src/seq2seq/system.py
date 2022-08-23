from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.nn import MSELoss, L1Loss
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import Seq2seqDataset, ExpandedSeq2SeqDataset
from .model import Encoder, Decoder
from src.base import BaseDataModule


class Seq2seqSystem(pl.LightningModule):
    def __init__(self, input_dim: int = 26, output_dim: int = 164):
        super(Seq2seqSystem, self).__init__()
        self.encoder = Encoder(input_dim, 150, 2)
        self.decoder = Decoder(output_dim, 150, 300)
        self.mse = MSELoss()
        self.l1 = L1Loss()
        self.alpha = 0.01
        self.beta = 1.0
        self.gamma = 0.5

    def forward(self, x, p):
        output, hidden = self.encoder(x)
        predicted_poses = self.decoder(output, hidden, p)
        return predicted_poses

    def custom_loss2(self, output, target,history, stage='train'):
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)
        history = history.transpose(0, 1)

        mse_loss = self.mse(output, target)
        self.log(f'{stage}/mse_loss', mse_loss)

        output_velocity = output[:, 1:, :] - output[:, :-1, :]
        target_velocity = target[:, 1:, :] - target[:, :-1, :]
        velocity_loss = self.mse(output_velocity, target_velocity)
        self.log(f'{stage}/velocity_loss', velocity_loss)

        loss = mse_loss + 0.6 * velocity_loss

        self.log(f'{stage}/loss', loss)

        return loss

    def custom_loss(self, output, target, history, stage='train'):
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)
        history = history.transpose(0, 1)

        n_element = output.numel()
        # MSE
        l1_loss = self.l1(output, target)
        self.log(f'{stage}/l1_loss', l1_loss)

        # continuous motion
        diff = [abs(output[:, n, :] - output[:, n - 1, :]) for n in range(1, output.shape[1])]
        cont_loss = torch.sum(torch.stack(diff)) / n_element
        self.log(f'{stage}/cont_loss', cont_loss)

        # history loss
        hist_loss = self.l1(output[:, 0, :], history[:, -1, :])
        self.log(f'{stage}/hist_loss', hist_loss)

        # motion variance
        norm = torch.norm(output, 2, 1)
        var_loss = -torch.sum(norm) / n_element
        self.log(f'{stage}/var_loss', var_loss)

        loss = l1_loss + self.alpha * cont_loss + self.beta * var_loss + self.gamma * hist_loss
        self.log(f'{stage}/loss', loss)

        return loss

    def training_step(self, batch, batch_nb):
        x, y, p = batch

        # before autoregression
        if self.current_epoch < 20:
            pred_poses = self.forward(x, p.new_zeros(p.shape))
        else:
            pred_poses = self.forward(x, p)
        # teacher forcing
        # batch_size = p.shape[1]
        # mask = torch.rand(batch_size) > 0.5
        # mask = mask.to(self.device)
        # p = p.transpose(1, 2) * mask
        # p = p.transpose(1, 2)

        loss = self.custom_loss2(pred_poses, y, p, 'train')
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y, p = batch
        pred_poses = self.forward(x, p)
        loss = self.custom_loss2(pred_poses, y, p, 'val')
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class Seq2seqDataModule(pl.LightningDataModule):
    def __init__(self, trn_folder: str, val_folder: str, stride: int = 15, batch_size: int = 32):
        super().__init__()
        self.trn_folder = trn_folder
        self.val_folder = val_folder
        self.batch_size = batch_size
        self.stride = stride
        self.trn_dataset = None  # type: FeedforwardDataset
        self.val_dataset = None  # type: FeedforwardDataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.trn_dataset = Seq2seqDataset(Path(self.trn_folder).glob('*.npz'), self.stride)
        self.val_dataset = Seq2seqDataset(Path(self.val_folder).glob('*.npz'), self.stride)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)


class ExpandedSeq2seqDataModule(BaseDataModule):
    def __init__(self, trn_folder: str, val_folder: str, train_metadata_path: str, val_metadata_path: str,
                 filter_user: int = None, only_fingers: bool = False, stride: int = 15, batch_size: int = 32,
                 audio_framerate: float = 30.):
        super().__init__(trn_folder, val_folder, train_metadata_path, val_metadata_path, filter_user, only_fingers)
        self.batch_size = batch_size
        self.stride = stride
        self.audio_framerate = audio_framerate
        self.trn_dataset = None  # type: ExpandedSeq2SeqDataset
        self.val_dataset = None  # type: ExpandedSeq2SeqDataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.trn_dataset = ExpandedSeq2SeqDataset(self.train_samples, seq_len=30, history=15,
                                                  stride=self.stride, audio_framerate=self.audio_framerate)
        self.val_dataset = ExpandedSeq2SeqDataset(self.val_samples, seq_len=30, history=15,
                                                  stride=self.stride, audio_framerate=self.audio_framerate)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)