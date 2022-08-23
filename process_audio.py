import librosa
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import logging

import scipy.signal

from src.utils.spect import make_spect_for_autovc

SAMPLE_RATE = 44100
FPS = 30
HOP_LENGTH = SAMPLE_RATE // FPS


def audio2mfcc(audio_path: Path):
    wav, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(
        y=wav, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=HOP_LENGTH * 2, n_mfcc=26)
    return np.transpose(mfcc)


def audio2logmel(audio_path):
    wav, sr = librosa.load(str(audio_path))
    spect = librosa.feature.melspectrogram(wav, sr=SAMPLE_RATE, window=scipy.signal.hanning,
                                          hop_length=HOP_LENGTH, fmax=7500, fmin=100, n_mels=64)
    eps=1e-10
    log_spect = np.log(abs(spect)+eps)
    return np.transpose(log_spect)


def wav2features(data_path: Path, dst_dir: Path, mode: str='mfcc'):
    recordings = list(data_path.glob('*.wav')) if data_path.is_dir() else [data_path]
    if not dst_dir.exists():
        dst_dir.mkdir()

    for recording in recordings:
        logging.info(recording)
        # make features
        features = None
        if mode == 'mfcc':
            features = audio2mfcc(recording)
        elif mode == 'mel':
            features = make_spect_for_autovc(str(recording))
        elif mode == 'logmel':
            features = audio2logmel(recording)
        assert features is not None

        logging.info(f'Audio features shape: {features.shape}')
        dst_path = dst_dir / recording.name.replace('.wav', '.npy')
        np.save(str(dst_path), features)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', help='Path to audio folder')
    arg_parser.add_argument('--dst', help='Path to store results')
    arg_parser.add_argument('--mode', choices=['mfcc', 'mel', 'logmel'], default='mfcc', help='Type of audio encoding')
    args = arg_parser.parse_args()
    wav2features(Path(args.src), Path(args.dst), args.mode)

