import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\combined_eeg_dataset.csv")

target_col = "label"
group_col = "Subject"

X = df.drop(columns=[target_col, group_col])
y = df[target_col]
groups = df[group_col]

# Subject-wise split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["Before", "After"],
    mode='classification'
)

i = 5  # sample index

exp = explainer.explain_instance(
    X_test.iloc[i].values,
    pca_99_pipeline.predict_proba
)

exp.save_to_file("lime_explanation.html")
print("LIME saved!")