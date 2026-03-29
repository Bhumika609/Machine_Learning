# src/windowing.py

import numpy as np

def create_windows(data, fs=1000, window_sec=2, overlap=0.5):
    window_size = fs * window_sec
    step_size = int(window_size * (1 - overlap))

    windows = []

    for start in range(0, len(data) - window_size, step_size):
        window = data[start:start + window_size]
        windows.append(window)

    return windows
