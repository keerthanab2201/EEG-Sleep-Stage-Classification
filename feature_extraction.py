import numpy as np
from scipy import signal
from typing import Tuple

BAND_RANGES = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
}
BAND_NAMES = ["Delta", "Theta", "Alpha", "Beta"]


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.trapz(psd[mask], freqs[mask]))


def extract_features(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    features = []
    features.append(float(np.mean(epoch)))
    features.append(float(np.var(epoch)))
    features.append(float(np.sqrt(np.mean(epoch**2))))
    freqs, psd = signal.welch(
        epoch, fs=sfreq, nperseg=int(4 * sfreq), noverlap=int(2 * sfreq)
    )
    for bn in BAND_NAMES:
        lo, hi = BAND_RANGES[bn]
        features.append(band_power(freqs, psd, lo, hi))
    return np.array(features)


def extract_features_bulk(
    X: np.ndarray, y: np.ndarray, sfreq: float
) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    feature_list = []
    for i in range(n):
        feature_list.append(extract_features(X[i], sfreq))
    return np.array(feature_list), y.copy()


FEATURE_NAMES = ["Mean", "Variance", "RMS", "Delta", "Theta", "Alpha", "Beta"]
