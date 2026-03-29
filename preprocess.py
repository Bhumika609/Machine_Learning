# src/preprocess.py

import pandas as pd

def preprocess_file(filepath):
    df = pd.read_excel(filepath)

    # Keep first 19 columns only
    df = df.iloc[:, :19]

    return df
