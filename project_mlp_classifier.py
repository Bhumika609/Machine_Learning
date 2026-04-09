import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# =========================
# 1. PATHS
# =========================

# Combined dataset path
file_path = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\combined_eeg_dataset.csv"

# Main project folder
project_root = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection"

# Existing model_outputs folder
output_dir = os.path.join(project_root, "model_outputs")
os.makedirs(output_dir, exist_ok=True)

# Existing excel file where all model results are stored
excel_path = os.path.join(output_dir, "model_comparison_results.xlsx")

# =========================
# 2. LOAD DATASET
# =========================

if file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    df = pd.read_excel(file_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())

# =========================
# 3. DEFINE FEATURES, LABEL, GROUPS
# =========================

target_col = "label"
group_col = "Subject"

feature_cols = [col for col in df.columns if col not in [target_col, group_col]]

X = df[feature_cols]
y = df[target_col]
groups = df[group_col]

print("\nNumber of features:", X.shape[1])
print("Target classes:", y.unique())
print("Number of subjects:", groups.nunique())

# =========================
# 4. SUBJECT-WISE TRAIN TEST SPLIT
# =========================

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# 5. DEFINE MLP CLASSIFIER
# =========================

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)

# =========================
# 6. TRAIN MODEL
# =========================

print("\nTraining MLP Classifier...")
mlp_clf.fit(X_train, y_train)
print("Training completed.")

# =========================
# 7. PREDICTIONS
# =========================

y_train_pred = mlp_clf.predict(X_train)
y_test_pred = mlp_clf.predict(X_test)

# predict_proba needed for ROC-AUC
y_test_proba = mlp_clf.predict_proba(X_test)[:, 1]

# =========================
# 8. EVALUATION METRICS
# =========================

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

# =========================
# 9. FIT STATUS
# =========================

gap = train_acc - test_acc

if train_acc < 0.65 and test_acc < 0.65:
    fit_status = "Underfitting"
elif gap > 0.15:
    fit_status = "Overfitting"
else:
    fit_status = "Good Fit"

# =========================
# 10. PRINT RESULTS
# =========================

print("\n========== MLP CLASSIFIER RESULTS ==========")
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-score       : {f1:.4f}")
print(f"Specificity    : {specificity:.4f}")
print(f"ROC-AUC        : {roc_auc:.4f}")
print(f"Fit Status     : {fit_status}")

print("\nConfusion Matrix:")
print(cm)

# =========================
# 11. SAVE CONFUSION MATRIX IMAGE
# =========================

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Before Workload', 'After Workload'],
            yticklabels=['Before Workload', 'After Workload'])
plt.title("Confusion Matrix - MLP")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

cm_path = os.path.join(output_dir, "MLP_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()

print(f"\nConfusion matrix saved to: {cm_path}")

# =========================
# 12. SAVE RESULT INTO EXISTING EXCEL
# =========================

mlp_result = pd.DataFrame([{
    "Model": "MLP",
    "Train Acc": round(train_acc, 4),
    "Test Acc": round(test_acc, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1-score": round(f1, 4),
    "Specificity": round(specificity, 4),
    "ROC-AUC": round(roc_auc, 4),
    "Fit Status": fit_status
}])

print("\nMLP Result Row:")
print(mlp_result)

if os.path.exists(excel_path):
    existing_df = pd.read_excel(excel_path)

    # Remove old MLP row if already present
    existing_df = existing_df[existing_df["Model"] != "MLP"]

    updated_df = pd.concat([existing_df, mlp_result], ignore_index=True)
    updated_df.to_excel(excel_path, index=False)

else:
    mlp_result.to_excel(excel_path, index=False)

print("\nMLP result successfully saved into existing file:")
print(excel_path)