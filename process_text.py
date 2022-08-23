from typing import Dict
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm
import json
from math import floor


def build_vocab(tsv_folder: Path) -> Dict[str, int]:
    vocab = {'PAD': 0, 'UNK': 1}
    unique_symbols = set()
    for tsv in tqdm(tsv_folder.glob('*.tsv')):
        with open(tsv, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                parts = line.split('\t')
                if len(parts) == 3:
                    _, _, word = parts
                elif len(parts) == 2:
                    word = ' '
                else:
                    logging.warning(f'Bad line: {line} in file {tsv}')
                    continue
                for s in word.lower():
                    unique_symbols.add(s)

    for s in unique_symbols:
        vocab[s] = len(vocab)
    return vocab


def process_tsv(tsv: Path, vocab: Dict[str, int], fps: float = 30.):
    starts, ends, words = [], [], []
    with open(tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split('\t')
            if len(parts) == 3:
                start, end, word = parts
            elif len(parts) == 2:
                start, end = parts
                word = ' '
            else:
                logging.warning(f'Bad line: {line} in file {tsv}')
                continue
            starts.append(float(start))
            ends.append(float(end))
            words.append(word.lower())

    if len(ends) == 0:
        return np.zeros(1)

    indexes = np.zeros(floor(fps * ends[-1]))
    for start, end, word in zip(starts, ends, words):
        word_start = floor(fps * start)
        word_end = floor(fps * end)
        word_len = word_end - word_start

        for i, s in enumerate(word):
            symbol_start = word_start + floor(i * word_len / len(word))
            symbol_end = min(word_start + floor((i + 1) * word_len / len(word)), word_end)
            symbol_idx = vocab[s] if s in vocab else vocab['UNK']
            indexes[symbol_start:symbol_end] = np.ones(symbol_end - symbol_start) * symbol_idx
    return indexes


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', help='Path to tsv folder')
    arg_parser.add_argument('--dst', help='Path to store results')
    arg_parser.add_argument('--vocab', help='Path to store or load vocab')
    arg_parser.add_argument('--fps', type=float, help='Audio features framerate', default=30.)
    args = arg_parser.parse_args()

    vocab_path = Path(args.vocab)
    src_path = Path(args.src)
    if not vocab_path.exists():
        logging.info('Building new vocab...')
        vocab = build_vocab(src_path)
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        logging.info('Loading vocab from file...')
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

    dst_path = Path(args.dst)
    if not dst_path.exists():
        dst_path.mkdir()

    weights = np.identity(len(vocab) - 1)
    weights = np.concatenate([np.zeros((1, len(weights))), weights], axis=0)

    for tsv in tqdm(src_path.glob('*.tsv')):
        indexes = process_tsv(tsv, vocab, args.fps)
        embeddings = weights[indexes.astype(int)]
        dst_sample = dst_path / tsv.name.replace('.tsv', '.npy')
        np.save(dst_sample, embeddings)

    # print(indexes)
