# src/feature_extraction.py

import numpy as np
from scipy.signal import welch

bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

def extract_features_from_window(window, fs=1000):
    features = []

    for channel in window.columns:
        signal = window[channel].values

        freqs, psd = welch(signal, fs=fs, nperseg=fs)

        for band in bands.values():
            low, high = band
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psd[idx], freqs[idx])
            features.append(band_power)

    return features
