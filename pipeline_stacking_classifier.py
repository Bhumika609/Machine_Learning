import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =========================
# 1. LOAD DATASET
# =========================

file_path = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\combined_eeg_dataset.csv"

if file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    df = pd.read_excel(file_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# =========================
# 2. FEATURES & LABELS
# =========================

target_col = "label"
group_col = "Subject"

feature_cols = [col for col in df.columns if col not in [target_col, group_col]]

X = df[feature_cols]
y = df[target_col]
groups = df[group_col]

# =========================
# 3. GROUP SPLIT
# =========================

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Train:", X_train.shape, "Test:", X_test.shape)

# =========================
# 4. BASE MODELS
# =========================

base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, kernel='rbf', random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=42)),
    ('cat', CatBoostClassifier(iterations=200, depth=5,
                               learning_rate=0.05, verbose=0, random_state=42))
]

# =========================
# 5. META MODELS
# =========================

meta_models = {
    "LR": LogisticRegression(max_iter=1000, random_state=42),
    "RF": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGB": XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=42)
}

# =========================
# 6. OUTPUT PATH
# =========================

project_root = os.path.dirname(os.path.dirname(__file__))
output_dir = os.path.join(project_root, "model_outputs")
os.makedirs(output_dir, exist_ok=True)

excel_path = os.path.join(output_dir, "model_comparision_results.xlsx")

all_results = []

# =========================
# 7. LOOP OVER META MODELS
# =========================

for name, meta_model in meta_models.items():

    print(f"\n===== Pipeline + Stacking with Meta Model: {name} =====")

    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        stack_method='predict_proba',
        cv=5,
        n_jobs=-1,
        passthrough=True, 
    )

    # 🔥 PIPELINE (Scaler + Stacking)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('stacking', stacking_clf)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    gap = train_acc - test_acc
    if train_acc < 0.65 and test_acc < 0.65:
        fit_status = "Underfitting"
    elif gap > 0.15:
        fit_status = "Overfitting"
    else:
        fit_status = "Good Fit"

    # =========================
    # SAVE CONFUSION MATRIX
    # =========================

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Pipeline Stacking ({name}) Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_path = os.path.join(output_dir, f"Pipeline_Stacking_{name}_cm.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Saved CM: {cm_path}")

    # =========================
    # STORE RESULTS
    # =========================

    all_results.append({
        "Model": f"Pipeline Stacking ({name})",
        "Train Acc": round(train_acc, 4),
        "Test Acc": round(test_acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "Specificity": round(specificity, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Fit Status": fit_status
    })

# =========================
# 8. SAVE TO EXISTING EXCEL
# =========================

new_results_df = pd.DataFrame(all_results)

if os.path.exists(excel_path):
    existing_df = pd.read_excel(excel_path)
    final_df = pd.concat([existing_df, new_results_df], ignore_index=True)
else:
    final_df = new_results_df

final_df.to_excel(excel_path, index=False)

print("\nAll pipeline stacking results saved to:")
print(excel_path)