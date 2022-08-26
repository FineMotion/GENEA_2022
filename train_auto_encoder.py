from argparse import ArgumentParser
from pathlib import Path
from src.auto_encoder import AutoEncoderSystem
from src.auto_encoder import AutoEncoderDataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import shutil
import logging
import random

random.seed(42)


class SystemSelector:
    @staticmethod
    def add_system_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--trn_folder', type=str, required=True)
        parser.add_argument('--val_folder', type=str, required=True)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--input_dim', type=int, default=164)
        parser.add_argument('--frames_count', type=int, default=3)
        parser.add_argument('--output_dim', type=int, default=60)
        parser.add_argument('--hidden_dim', type=int, default=512)
        return parser

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.system = None  # type: pl.LightningModule
        self.datamodule = None  # type: pl.LightningDataModule

    def initialize(self):

        # unpack kwargs to initialize system
        input_dim = self.kwargs['input_dim']
        frames_count = self.kwargs['frames_count']
        output_dim = self.kwargs['output_dim']
        hidden_dim = self.kwargs['hidden_dim']

        self.initialize_system(input_dim, frames_count, output_dim, hidden_dim)

        # unpack kwargs to initialize datamodule
        trn_folder = self.kwargs['trn_folder']
        val_folder = self.kwargs['val_folder']
        batch_size = self.kwargs['batch_size']

        self.initialize_datamodule(trn_folder=trn_folder, val_folder=val_folder, batch_size=batch_size)

    def initialize_system(self, input_dim: int = 164, frames_count: int = 3, output_dim: int = 60,
                          hidden_dim: int = 512):
        self.system = AutoEncoderSystem(input_dim, frames_count, output_dim, hidden_dim)

    def initialize_datamodule(self, trn_folder: str, val_folder: str, batch_size: int = 128):
        base_kwargs = {
            'trn_folder': trn_folder,
            'val_folder': val_folder,
            'batch_size': batch_size
        }

        self.datamodule = AutoEncoderDataModule(**base_kwargs)


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

    patience_callback = EarlyStopping(
        min_delta=0.0,
        mode='min',
        monitor='val/loss',
        patience=20
    )

    system_selector = SystemSelector(**vars(args))
    system_selector.initialize()

    trainer = Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback, patience_callback])
    trainer.fit(model=system_selector.system, datamodule=system_selector.datamodule)
