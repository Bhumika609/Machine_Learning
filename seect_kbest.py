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
from sklearn.feature_selection import SelectKBest, f_classif

fs_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=30)),
    ('model', LogisticRegression(max_iter=1000))
])

fs_pipeline.fit(X_train, y_train)

y_pred = fs_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Feature Selection Accuracy:", acc)