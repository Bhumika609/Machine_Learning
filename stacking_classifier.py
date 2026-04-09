import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
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

# Change this path to your combined dataset file
file_path = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\combined_eeg_dataset.csv"   # or .csv

if file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    df = pd.read_excel(file_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())

# =========================
# 2. DEFINE FEATURES, LABEL, GROUPS
# =========================

# Make sure your dataset has these columns exactly:
# Label -> class label
# Subject -> subject id

target_col = "label"
group_col = "Subject"

# Feature columns = everything except Label and Subject
feature_cols = [col for col in df.columns if col not in [target_col, group_col]]

X = df[feature_cols]
y = df[target_col]
groups = df[group_col]

print("\nNumber of features:", X.shape[1])
print("Target classes:", y.unique())
print("Number of subjects:", groups.nunique())

# =========================
# 3. SUBJECT-WISE TRAIN TEST SPLIT
# =========================

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# 4. DEFINE BASE MODELS
# =========================

base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )),
    
    ('ada', AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    )),
    
    ('svm', SVC(
        probability=True,
        kernel='rbf',
        random_state=42
    )),
    
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500,
        random_state=42
    )),
    
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
# 5. DEFINE META MODEL
# =========================

meta_model = LogisticRegression(max_iter=1000, random_state=42)

# =========================
# 6. BUILD STACKING CLASSIFIER
# =========================

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='predict_proba',
    cv=5,
    passthrough=False,
    n_jobs=-1
)

# =========================
# 7. TRAIN MODEL
# =========================

print("\nTraining Stacking Classifier...")
stacking_clf.fit(X_train, y_train)
print("Training completed.")

# =========================
# 8. PREDICTIONS
# =========================

y_train_pred = stacking_clf.predict(X_train)
y_test_pred = stacking_clf.predict(X_test)

y_test_proba = stacking_clf.predict_proba(X_test)[:, 1]

# =========================
# 9. EVALUATION METRICS
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
# 10. FIT STATUS
# =========================

gap = train_acc - test_acc

if train_acc < 0.65 and test_acc < 0.65:
    fit_status = "Underfitting"
elif gap > 0.15:
    fit_status = "Overfitting"
else:
    fit_status = "Good Fit"

# =========================
# 11. PRINT RESULTS
# =========================

print("\n========== STACKING CLASSIFIER RESULTS ==========")
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
# 12. SAVE CONFUSION MATRIX IMAGE
# =========================

output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Before Workload', 'After Workload'],
            yticklabels=['Before Workload', 'After Workload'])
plt.title("Confusion Matrix - Stacking Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

cm_path = os.path.join(output_dir, "Stacking_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()

print(f"\nConfusion matrix saved to: {cm_path}")

# =========================
# 13. SAVE RESULT ROW
# =========================

stacking_result = pd.DataFrame([{
    "Model": "Stacking Classifier",
    "Train Acc": round(train_acc, 4),
    "Test Acc": round(test_acc, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1-score": round(f1, 4),
    "Specificity": round(specificity, 4),
    "ROC-AUC": round(roc_auc, 4),
    "Fit Status": fit_status
}])

print("\nStacking Result Row:")
print(stacking_result)

# Save single result
stacking_result.to_excel(os.path.join(output_dir, "stacking_result.xlsx"), index=False)

print("\nStacking result saved successfully.")