from typing import Iterable

import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import logging
from tqdm import tqdm

from src.utils.normalization import create_motion_array, get_normalization_values, normalize_data, denormalize_data

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src')
    arg_parser.add_argument('--dst')
    arg_parser.add_argument('--normalization_values', type=str, help='Path to normalization values to load or store',
                            required=True)
    arg_parser.add_argument('--mode', choices=['forward', 'backward'], default='forward')
    args = arg_parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    src_path = Path(args.src)
    src_files = list(src_path.glob('*.npy')) if src_path.is_dir() else [src_path]

    normalization_values = Path(args.normalization_values)
    if normalization_values.exists():
        logging.info('Loading existing normalization values')
        normalization_data = np.load(normalization_values)
        max_val, mean_val = normalization_data['max_val'], normalization_data['mean_val']
    else:
        logging.info('Creating normalization values')
        all_data = create_motion_array(src_files)
        max_val, mean_val = get_normalization_values(all_data)
        np.savez(normalization_values, max_val=max_val, mean_val=mean_val)

    dst_path = Path(args.dst)
    if not dst_path.exists():
        dst_path.mkdir()

    for src_file in tqdm(src_files):
        dst_file = dst_path / src_file.name
        src_data = np.load(src_file)
        dst_data = normalize_data(src_data, max_val, mean_val) if args.mode == 'forward' \
            else denormalize_data(src_data, max_val, mean_val)
        np.save(dst_file, dst_data)


