# src/build_dataset.py

import os
import pandas as pd
import numpy as np

from src.preprocess import preprocess_file
from src.windowing import create_windows
from src.feature_extraction import extract_features_from_window

def build_dataset(raw_folder, save_folder):

    feature_folder = os.path.join(save_folder, "features")
    os.makedirs(feature_folder, exist_ok=True)

    for file in os.listdir(raw_folder):

        if file.endswith(".xlsx"):
            filepath = os.path.join(raw_folder, file)

            print("Processing:", file)

            # Assign label
            if "_1" in file:
                label = 0
            elif "_2" in file:
                label = 1
            else:
                continue

            # Preprocess (remove last 2 columns)
            df = preprocess_file(filepath)

            # Create windows
            windows = create_windows(df)

            all_features = []

            # Extract features per window
            for window in windows:
                features = extract_features_from_window(window)
                features.append(label)   # Add label at end
                all_features.append(features)

            # Create column names
            channel_names = df.columns
            band_names = ["delta", "theta", "alpha", "beta", "gamma"]

            columns = []
            for ch in channel_names:
                for band in band_names:
                    columns.append(f"{ch}_{band}")

            columns.append("label")

            # Convert to DataFrame
            feature_df = pd.DataFrame(all_features, columns=columns)

            # Save as Excel
            save_path = os.path.join(feature_folder, file.replace(".xlsx", "_features.xlsx"))
            feature_df.to_excel(save_path, index=False)

            print("Saved:", save_path)

    print("All feature files saved successfully.")
