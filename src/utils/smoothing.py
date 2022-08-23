import numpy as np
from scipy.signal import savgol_filter


def smoothing(motion):
    smoothed = [savgol_filter(motion[:, i], 9, 3) for i in range(motion.shape[1])]
    smoothed = np.array(smoothed).transpose()
    return smoothed
