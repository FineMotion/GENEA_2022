from pathlib import Path
from typing import Iterable

import numpy as np


def create_motion_array(data_files: Iterable[Path]):
    result = []
    for data_file in data_files:
        data = np.load(data_file)
        result.append(data)
    result = np.concatenate(result, axis=0)
    return result


def get_normalization_values(data: np.ndarray):
    max_val = np.amax(np.absolute(data), axis=0)
    mean_val = data.mean(axis=0)
    return max_val, mean_val


def normalize_data(data, max_val, mean_val, eps=1e-8):
    data_centered = data - mean_val[np.newaxis, :]
    data_normalized = np.divide(data_centered, max_val[np.newaxis, :] + eps)
    return data_normalized


def denormalize_data(data, max_val, mean_val, eps=1e-8):
    data_reconstructed = np.multiply(data, max_val[np.newaxis, :] + eps)
    data_reconstructed += mean_val[np.newaxis, :]
    return data_reconstructed
