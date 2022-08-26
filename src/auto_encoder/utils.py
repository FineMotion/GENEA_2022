import numpy as np
import os
import torch
from pathlib import Path
from srs.system import AutoEncoderSystem


def get_data_by_file(data_path: str):
    samples = list(Path(data_path).glob('*.npy'))
    processed_samples = []

    for i, sample in samples:
        processed_samples.append(np.load(sample))

    data = np.concatenate(processed_samples)

    return data


def normalize(data, min_values, max_values):
    return (data - min_values) / (max_values - min_values)


def denormalize(data, min_values, max_values):
    return data * (max_values - min_values) + min_values


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def encode_files(data_path, auto_enc_path, auto_enc_out_path, frames_to_encoder=3):
    auto_enc_system = AutoEncoderSystem.load_from_checkpoint(checkpoint_path=auto_enc_path)
    auto_enc_system.eval()

    with open(os.path.join(auto_enc_path, 'min_max.txt'), 'r') as f:
        min_val, max_val = f.read().split()
        min_val = int(min_val)
        max_val = int(max_val)

    samples_dir = list(Path(data_path).iterdir())

    for i, sample in enumerate(samples_dir):
        file_name = os.path.basename(sample)
        for file in os.listdir(sample):
            inpt = np.load(str(os.path.join(sample, file)))
            out = normalize(torch.tensor(inpt), min_val, max_val)
            out_samples = []
            for j in range(frames_to_encoder):
                out_samples.append(out[j: j + (len(out) - frames_to_encoder)])

            out = torch.cat(out_samples, axis=-1)
            out = auto_enc_system.encoder(out).cpu().detach().numpy()

            make_dir(auto_enc_out_path)
            np.save(os.path.join(auto_enc_out_path, file_name + "_" + str(file)), out)


def get_right_form(model_out, frames_to_encoder=3):
    second_shape = model_out.shape[1] // frames_to_encoder
    right_form = np.zeros((len(model_out) + frames_to_encoder - 1, second_shape))

    right_form[:len(model_out)] = model_out[:, :second_shape]

    for i in range(1, frames_to_encoder):
        right_form[i:i + len(model_out)] += model_out[:, i * second_shape:(i + 1) * second_shape]

    for i in range(frames_to_encoder - 1):
        right_form[i] /= (i + 1)
        right_form[-i - 1] /= (i + 1)

    right_form[frames_to_encoder - 1:-(frames_to_encoder - 1)] /= frames_to_encoder
    return right_form


def decode_pred(predictions, auto_enc_path):
    auto_enc_system = AutoEncoderSystem.load_from_checkpoint(checkpoint_path=auto_enc_path)
    auto_enc_system.eval()
    out = auto_enc_system.decoder(predictions.double()).cpu().detach().numpy()
    out = get_right_form(out)

    with open(os.path.join(auto_enc_path, 'min_max.txt'), 'r') as f:
        min_val, max_val = f.read().split()
        min_val = int(min_val)
        max_val = int(max_val)

    return denormalize(out, min_val, max_val)
