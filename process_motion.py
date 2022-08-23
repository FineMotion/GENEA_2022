from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from pymo.parsers import BVHParser
from pymo.writers import BVHWriter
from sklearn.pipeline import Pipeline
import joblib as jl
from tqdm import tqdm
import logging

from pymo.preprocessing import MocapParameterizer, ConstantsRemover, Numpyfier, RootTransformer


def generate_pipeline(data_path: Path, pipeline_dir: Path):
    bvh_parser = BVHParser()
    data = []
    bvh_names = list(data_path.glob('*.bvh')) if data_path.is_dir() else [data_path]
    for bvh_path in bvh_names:
        logging.info(bvh_path)
        data.append(bvh_parser.parse(str(bvh_path)))

    if not pipeline_dir.exists():
        pipeline_dir.mkdir()

    pipeline_path = pipeline_dir / 'data_pip.sav'
    data_pipe = Pipeline([
        ('root', RootTransformer('hip_centric')),
        ('exp', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())
    ])
    logging.info('Fitting pipeline...')
    data_pipe.fit(data)
    logging.info('Saving pipeline...')
    jl.dump(data_pipe, str(pipeline_path))


def bvh2features(data_path: Path, dst_dir: Path, pipeline_dir: Path):
    bvh_parser = BVHParser()

    pipeline_path = pipeline_dir / 'data_pip.sav'
    assert pipeline_path.exists()

    logging.info('Loading pipeline...')
    data_pipe = jl.load(str(pipeline_path))  # type: Pipeline

    logging.info('Transforming data...')
    bvh_names = list(data_path.glob('*.bvh')) if data_path.is_dir() else [data_path]
    dst_dir = Path(dst_dir)
    if not dst_dir.exists():
        dst_dir.mkdir()

    for bvh_path in tqdm(bvh_names):
        data = bvh_parser.parse(str(bvh_path))
        out_data = data_pipe.transform([data])[0]
        logging.info(f'Output shape: {out_data.shape}')

        dst_path = dst_dir / bvh_path.name.replace('.bvh', '.npy')
        logging.info(dst_path)
        np.save(str(dst_path), out_data)


def features2bvh(data_path: Path, dst_dir: Path, pipeline_dir: Path):
    logging.info('Loading pipeline...')
    pipeline_path = Path(pipeline_dir) / 'data_pip.sav'
    pipeline = jl.load(str(pipeline_path))  # type: Pipeline

    data_path = Path(data_path)
    recordings = list(data_path.glob('*.npy')) if data_path.is_dir() else [data_path]

    data = []
    for recording in recordings:
        features = np.load(str(recording))
        logging.info(f"{recording} motion features shape: {features.shape}")
        data.append(features)

    logging.info("Transforming data...")
    bvh_data = pipeline.inverse_transform(data)

    dst_dir = Path(dst_dir)
    if not dst_dir.exists(): dst_dir.mkdir()
    bvh_writer = BVHWriter()
    logging.info("Saving bvh...")
    for i, recording in enumerate(recordings):
        dst_path = dst_dir / recording.name.replace('.npy', '.bvh')
        logging.info(dst_path)
        with open(str(dst_path), 'w') as f:
            bvh_writer.write(bvh_data[i], f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', help='Path to data folder')
    arg_parser.add_argument('--dst', help='Path to folder to store results')
    arg_parser.add_argument('--pipeline_dir', default='./pipe', help='Pth to save pipeline')
    arg_parser.add_argument('--mode', choices=['bvh2npy', 'npy2bvh', 'pipeline'], help='Modes of processing',
                            default='bvh2npy')
    args = arg_parser.parse_args()

    if args.mode == 'pipeline':
        generate_pipeline(Path(args.src), Path(args.pipeline_dir))
    elif args.mode == 'npy2bvh':
        features2bvh(Path(args.src), Path(args.dst), Path(args.pipeline_dir))
    elif args.mode == 'bvh2npy':
        bvh2features(Path(args.src), Path(args.dst), Path(args.pipeline_dir))
    else:
        logging.warning(f'Unsupported mode: {args.mode}')
