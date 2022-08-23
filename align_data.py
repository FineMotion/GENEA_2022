from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import logging
import math


def align(motion_path: Path, audio_path: Path, text_path: Path, dst_dir: Path, mode: str):
    # for test samples
    logging.info('Processing test samples, no motion data...')
    motion_files = None
    if motion_path is not None:
        motion_files = set(motion_path.glob('*.npy')) if motion_path.is_dir() else {motion_path}

    audio_files = set(audio_path.glob('*.npy')) if audio_path.is_dir() else {audio_path}
    text_files = set(text_path.glob('*.npy')) if text_path.is_dir() else {text_path}

    if not dst_dir.exists():
        dst_dir.mkdir()

    for audio_file in audio_files:
        text_file = text_path / audio_file.name
        motion_file = audio_path / audio_file.name if motion_files is not None else None

        if motion_file is not None and motion_file not in motion_files:
            logging.warning(f'Missing motion file: {audio_file.name}')
        if text_file not in text_files:
            logging.warning(f'Missing text file: {audio_file.name}')

        logging.info(audio_file.name)

        audio = np.load(str(audio_file))
        text = np.load(str(text_file))

        if motion_file is None:
            audio_len = audio.shape[1] if mode == 'mfcc' else audio.shape[0]
            motion_len = audio_len if mode == 'mfcc' else math.floor(audio_len * 30. / 62.5)
            motion = np.zeros((motion_len, 164), dtype=float)
        else:
            motion = np.load(str(motion_file))

        if mode in {'mfcc', 'logmel'}:
            min_len = min(motion.shape[0], audio.shape[0])
            motion_len, audio_len = min_len, min_len
        elif mode == 'mel':
            motion_len = motion.shape[0]
            audio_len = math.floor(motion_len * 62.5 / 30.)
            assert audio_len <= audio.shape[0]
        else:
            assert False
        text_paddings = np.zeros((audio.shape[0] - text.shape[0], text.shape[1]))
        text = np.concatenate([text, text_paddings], axis=0)

        input_features = np.concatenate([audio, text], axis=1)
        input_features = input_features[:audio_len]

        motion = motion[:motion_len]
        logging.info(f'Output shape: {motion.shape}')
        logging.info(f'Input shape: {input_features.shape}')
        result_path = dst_dir / audio_file.name.replace('.npy', '.npz')
        np.savez(str(result_path), X=input_features, Y=motion)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--motion_dir', help='Path to motion data')
    arg_parser.add_argument('--audio_dir', help='Path to audio data')
    arg_parser.add_argument('--text_dir', help='Path to text data')
    arg_parser.add_argument('--dst', help='Path to store results')
    arg_parser.add_argument('--mode', choices=['mfcc', 'mel', 'logmel'], default='mfcc')
    args = arg_parser.parse_args()

    align(Path(args.motion_dir) if args.motion_dir is not None else None,
          Path(args.audio_dir), Path(args.text_dir), Path(args.dst), args.mode)
