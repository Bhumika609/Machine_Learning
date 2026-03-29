# src/load_data.py

import os
import pandas as pd

def load_all_files(folder_path):
    files = []

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            full_path = os.path.join(folder_path, file)
            files.append((file, full_path))

    return files
