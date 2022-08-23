import pytorch_lightning as pl
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import joblib as jl

from pymo.writers import BVHWriter
from src.utils.smoothing import smoothing
from src.utils.normalization import denormalize_data


class Predictor:
    def __init__(self, model_name, input_dim, output_dim, window_size):
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.system = None  # type: pl.LightningModule
        self.dataset = None  # type: Dataset

    def initialize_models(self, checkpoint_path, data_path):
        if self.model_name == 'feedforward':
            from src.feedforward import FeedforwardSystem, FeedforwardDataset
            self.system = FeedforwardSystem(in_features=self.input_dim, out_features=self.output_dim)
            self.dataset = FeedforwardDataset([data_path])
        elif self.model_name == 'recell':
            from src.recell import ReCellSystem, ReCellDataset
            self.system = ReCellSystem(self.input_dim, self.output_dim, self.window_size)
            self.dataset = ReCellDataset([data_path], window_size=self.window_size)
        elif self.model_name == 'lstm':
            from src.lstm import LstmSystem, LstmDataset
            self.system = LstmSystem(in_features=self.input_dim, out_features=self.output_dim)
            self.dataset = LstmDataset([data_path], stride=60, window_size=60)
        elif self.model_name == 'seq2seq':
            from src.seq2seq import Seq2seqSystem, Seq2seqDataset, ExpandedSeq2SeqDataset
            self.system = Seq2seqSystem(input_dim=self.input_dim, output_dim=self.output_dim)
            self.dataset = Seq2seqDataset([data_path], stride=30)
        elif self.model_name == 'wav2gest':
            from src.wav2gest import Wav2GestSystem, Wav2GestDataset
            self.system = Wav2GestSystem(input_dim=self.input_dim, output_dim=self.output_dim)
            self.dataset = Wav2GestDataset([data_path], stride=60)
        assert self.system is not None
        assert self.dataset is not None
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.system.load_state_dict(checkpoint["state_dict"])
        self.system.eval()

    def predict(self):
        dataset_iter = iter(self.dataset)
        all_predictions = []

        h = None
        if self.model_name in {'seq2seq', 'wav2gest'}:
            h = np.zeros((15, self.output_dim), dtype=float)
            h = torch.FloatTensor(h)
            h = h.unsqueeze(0) if args.model == 'wav2gest' else h.unsqueeze(1)
        elif self.model_name == 'recell':
            h = np.zeros(self.output_dim, dtype=float)
            h = torch.FloatTensor(h)
            h = h.unsqueeze(0)

        with torch.no_grad():
            for i, sample in tqdm(enumerate(dataset_iter)):

                x = sample[0]
                x = x.unsqueeze(1) if args.model == 'seq2seq' else x.unsqueeze(0)
                pred = self.system(x, h) if args.model in {'seq2seq', 'wav2gest', 'recell'} else self.system(x)

                if args.model == 'seq2seq':
                    all_predictions.append(pred.squeeze(1).detach().cpu().numpy())
                else:
                    all_predictions.append(pred.squeeze(0).detach().cpu().numpy())
                if args.model in {'seq2seq', 'wav2gest'}:
                    h = pred[:, -15:, :]
                elif args.model == 'recell':
                    h = pred

        all_predictions = np.stack(all_predictions, 0) if args.model in {'feedforward', 'recell'} \
            else np.concatenate(all_predictions, 0)
        print(f'Prediction shape: {all_predictions.shape}')
        return all_predictions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--src", type=str, required=True, help="Path to input npz file with audio data")
    parser.add_argument("--dst", type=str, required=True, help="Path to store result npy with generated motion")
    parser.add_argument("--model", type=str, choices=['wav2gest', 'recell', 'feedforward', 'seq2seq', 'lstm'])
    # defaults
    parser.add_argument("--input_dim", type=int, default=26)
    parser.add_argument("--output_dim", type=int, default=164)
    parser.add_argument("--pipeline_dir", type=str, default='./pipe_v1')
    parser.add_argument("--window_size", type=int, default=61)
    args = parser.parse_args()

    # predict via network
    predictor = Predictor(args.model, args.input_dim, args.output_dim, args.window_size)
    predictor.initialize_models(args.checkpoint, args.src)
    predictions = predictor.predict()

    # smooth predictions
    smoothed = smoothing(predictions)

    # denormalize
    pipeline_dir = Path(args.pipeline_dir)
    normalization_values = pipeline_dir / 'normalization_values.npz'
    assert normalization_values.exists()
    normalization_data = np.load(normalization_values)
    max_val, mean_val = normalization_data['max_val'], normalization_data['mean_val']
    normalized = denormalize_data(smoothed, max_val, mean_val)

    # make bvh
    pipeline_path = pipeline_dir / 'data_pip.sav'
    pipeline = jl.load(str(pipeline_path))  # type: Pipeline
    bvh_data = pipeline.inverse_transform([normalized])
    bvh_writer = BVHWriter()
    with open(args.dst, 'w') as f:
        bvh_writer.write(bvh_data[0], f)
