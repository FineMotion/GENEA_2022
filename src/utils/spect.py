from pathlib import Path
from typing import Union

import numpy as np
from scipy import signal
from scipy.signal import get_window
import soundfile as sf
from librosa.filters import mel
from numpy.random import RandomState


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)


def make_spect_for_autovc(audio_path: Union[str, Path]):
    x, fs = sf.read(audio_path)
    # assume fs = 16kHz
    b, a = butter_highpass(30, fs, order=5)
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))

    y = signal.filtfilt(b, a, x)
    prng = RandomState(0)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5)*1e-06
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)
    return S.astype(np.float32)