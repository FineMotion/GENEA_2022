from argparse import ArgumentParser
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import shutil
import numpy as np
import logging
import random

random.seed(42)


class SystemSelector:
    @staticmethod
    def add_system_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--trn_folder', type=str, required=True)
        parser.add_argument('--val_folder', type=str, required=True)
        parser.add_argument('--model', type=str, choices=[
            'wav2gest', 'recell', 'recellseq', 'feedforward', 'seq2seq', 'lstm'], required=True)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--train_metadata', type=str, default=None)
        parser.add_argument('--val_metadata', type=str, default=None)
        parser.add_argument('--filter_user', type=int, default=None)
        parser.add_argument('--only_fingers', action="store_true")
        parser.add_argument('--window_size', type=int, default=125)
        parser.add_argument('--audio_framerate', type=float, default=30.)
        parser.add_argument('--seq_len', type=int, default=10)
        parser.add_argument('--stride', type=int, default=5)
        return parser

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = kwargs.get('model')
        assert self.model_name is not None

        trn_folder = Path(kwargs.get('trn_folder'))
        print(trn_folder)
        assert trn_folder.exists()
        sample = np.load(next(trn_folder.glob('*.npz')))
        self.input_dim = sample['X'].shape[1]
        self.output_dim = sample['Y'].shape[1]
        logging.info(f'Input dim: {self.input_dim}\tOutput dim: {self.output_dim}')

        self.system = None  # type: pl.LightningModule
        self.datamodule = None  # type: pl.LightningDataModule

    def initialize(self):
        self.initialize_system()

        # unpack kwargs to initialize datamodule
        trn_folder = self.kwargs['trn_folder']
        val_folder = self.kwargs['val_folder']
        batch_size = self.kwargs['batch_size']
        train_metadata_path = self.kwargs['train_metadata']
        val_metadata_path = self.kwargs['val_metadata']
        filter_user = self.kwargs.get('filter_user')
        only_fingers = self.kwargs.get('only_fingers')
        window_size = self.kwargs['window_size']
        audio_framerate = self.kwargs['audio_framerate']
        stride = self.kwargs['stride']
        self.initialize_datamodule(trn_folder=trn_folder, val_folder=val_folder,
                                   train_metadata_path=train_metadata_path, val_metadata_path=val_metadata_path,
                                   filter_user=filter_user, only_fingers=only_fingers, batch_size=batch_size, stride=stride,
                                   window_size=window_size, audio_framerate=audio_framerate)

    def initialize_system(self):
        if self.model_name == 'feedforward':
            from src.feedforward import FeedforwardSystem
            self.system = FeedforwardSystem(in_features=self.input_dim, out_features=self.output_dim)
        elif self.model_name == 'lstm':
            from src.lstm import LstmSystem
            self.system = LstmSystem(in_features=self.input_dim, out_features=self.output_dim)
        elif self.model_name == 'seq2seq':
            from src.seq2seq import Seq2seqSystem
            self.system = Seq2seqSystem(input_dim=self.input_dim, output_dim=self.output_dim)
        elif self.model_name == 'wav2gest':
            from src.wav2gest import Wav2GestSystem
            self.system = Wav2GestSystem(input_dim=self.input_dim, output_dim=self.output_dim)
        elif self.model_name == 'recell':
            from src.recell import ReCellSystem
            self.system = ReCellSystem(input_dim=self.input_dim, output_dim=self.output_dim)
        elif self.model_name == 'recellseq':
            from src.recell import ReCellSeqSystem
            window_size = self.kwargs['window_size']
            self.system = ReCellSeqSystem(input_dim=self.input_dim, output_dim=self.output_dim, window_size=window_size)
        assert self.system is not None

    def initialize_datamodule(self, trn_folder: str, val_folder: str, train_metadata_path: str = None,
                              val_metadata_path: str = None, filter_user: int = None, only_fingers: bool = False,
                              batch_size: int = 128, stride: int = 5, window_size: int = 61,
                              audio_framerate: float = 30., seq_len: int = 10):

        base_kwargs = {
            'trn_folder': trn_folder,
            'val_folder': val_folder,
            'train_metadata_path': train_metadata_path,
            'val_metadata_path': val_metadata_path,
            'filter_user': filter_user,
            'only_fingers': only_fingers,
            'batch_size': batch_size
        }

        if self.model_name == 'feedforward':
            from src.feedforward import FeedforwardDataModule
            self.datamodule = FeedforwardDataModule(**base_kwargs)
        elif self.model_name == 'lstm':
            from src.lstm import LstmDataModule
            lstm_kwargs = {
                **base_kwargs,
                'stride': stride,
                'window_size': window_size
            }
            self.datamodule = LstmDataModule(**lstm_kwargs)
        elif self.model_name == 'recell':
            from src.recell import ReCellDataModule
            self.datamodule = ReCellDataModule(**base_kwargs)
        elif self.model_name == 'recellseq':
            from src.recell import ReCellSeqDataModule
            recellseq_kwargs = {
                **base_kwargs,
                'stride': stride,
                'audio_framerate': audio_framerate,
                'window_size': window_size,
                'seq_len': seq_len
            }
            logging.info(f'Training ReCellSeq sustem with params: {recellseq_kwargs}')
            self.datamodule = ReCellSeqDataModule(**recellseq_kwargs)
        elif self.model_name == 'seq2seq':
            from src.seq2seq import ExpandedSeq2seqDataModule
            seq2seq_kwargs = {
                **base_kwargs,
                'stride': stride,
                'audio_framerate': audio_framerate
            }
            self.datamodule = ExpandedSeq2seqDataModule(**seq2seq_kwargs)
        elif self.model_name == 'wav2gest':
            from src.wav2gest import Wav2GestDataModule
            wav2gest_kwargs = {
                **base_kwargs,
                'stride': stride,
                'audio_framerate': audio_framerate
            }
            self.datamodule = Wav2GestDataModule(**wav2gest_kwargs)
        assert self.datamodule is not None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--serialize_dir', type=str, required=True)
    arg_parser.add_argument("--force", action="store_true")
    arg_parser = SystemSelector.add_system_args(arg_parser)
    arg_parser = Trainer.add_argparse_args(arg_parser)
    args = arg_parser.parse_args()
    print(Path(args.serialize_dir))
    if Path(args.serialize_dir).exists():
        if args.force:
            logging.warning(f"Force flag activated. Deleting {args.serialize_dir}...")
            shutil.rmtree(args.serialize_dir)
        else:
            logging.error(f"{args.serialize_dir} already exists! Choose another folder or use --force to overwrite")
            exit(-1)

    Path(args.serialize_dir).mkdir(parents=True)
    wandb_logger = WandbLogger(name=Path(args.serialize_dir).name, project='genea2022')
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.serialize_dir,
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    # patience_callback = EarlyStopping(
    #     min_delta=0.0,
    #     mode='min',
    #     monitor='val/loss',
    #     patience=35
    # )
    system_selector = SystemSelector(**vars(args))
    system_selector.initialize()

    trainer = Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model=system_selector.system, datamodule=system_selector.datamodule)
