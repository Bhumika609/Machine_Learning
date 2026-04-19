import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # no GUI issues
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from lime.lime_tabular import LimeTabularExplainer

# =========================
# 1. LOAD DATA
# =========================

file_path = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\combined_eeg_dataset.csv"

df = pd.read_csv(file_path)

target_col = "label"
group_col = "Subject"

feature_cols = [col for col in df.columns if col not in [target_col, group_col]]

X = df[feature_cols]
y = df[target_col]
groups = df[group_col]

# =========================
# 2. SUBJECT-WISE SPLIT
# =========================

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Train:", X_train.shape, "Test:", X_test.shape)

# =========================
# 3. BASE MODELS
# =========================

base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, kernel='rbf', random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
    ('xgb', XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )),
    ('cat', CatBoostClassifier(
        iterations=200,
        depth=5,
        learning_rate=0.05,
        verbose=0,
        random_state=42
    ))
]

# =========================
# 4. BEST META MODEL (LR)
# =========================

meta_model = LogisticRegression(max_iter=1000, random_state=42)

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='predict_proba',
    cv=5,
    passthrough=True,   # you enabled this ✔
    n_jobs=-1
)

# =========================
# 5. PIPELINE
# =========================

pipeline_model = Pipeline([
    ('scaler', StandardScaler()),
    ('stack', stacking)
])

print("\nTraining Pipeline Stacking Model...")
pipeline_model.fit(X_train, y_train)
print("Training Done.")

# =========================
# 6. CREATE LIME EXPLAINER
# =========================

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_cols,
    class_names=["Before Workload", "After Workload"],
    mode='classification'
)

# =========================
# 7. OUTPUT DIRECTORY
# =========================

output_dir = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\model_outputs"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 8. GENERATE EXPLANATIONS
# =========================

# choose few test samples
sample_indices = [0, 10, 25, 50, 100]

for i in sample_indices:
    print(f"\nExplaining sample index: {i}")

    exp = explainer.explain_instance(
        data_row=X_test.iloc[i].values,
        predict_fn=pipeline_model.predict_proba,
        num_features=10
    )

    # Save plot
    fig = exp.as_pyplot_figure()
    save_path = os.path.join(output_dir, f"LIME_Explanation_{i}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved LIME explanation to: {save_path}")

print("\nAll LIME explanations generated successfully.")