import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

# Optional models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# -----------------------------------
# 0. CREATE OUTPUT FOLDER
# -----------------------------------
output_folder = "model_outputs"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------------
# 1. LOAD DATA
# -----------------------------------
df = pd.read_csv("combined_eeg_dataset.csv")

# Remove unwanted unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# -----------------------------------
# 2. DEFINE FEATURES / LABEL / GROUPS
# -----------------------------------
X = df.drop(columns=["Subject", "label"])
y = df["label"]
groups = df["Subject"]

# -----------------------------------
# 3. SUBJECT-WISE TRAIN-TEST SPLIT
# -----------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# -----------------------------------
# 4. DEFINE MODELS
# -----------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_state=42
    ),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,   # Needed for ROC-AUC
            random_state=42
        ))
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        ))
    ])
}

# -----------------------------------
# 5. FUNCTION TO CHECK FIT STATUS
# -----------------------------------
def check_fit_status(train_acc, test_acc):
    gap = train_acc - test_acc

    if train_acc < 0.70 and test_acc < 0.70:
        return "Underfitting"
    elif gap > 0.12:
        return "Overfitting"
    else:
        return "Good Fit"

# -----------------------------------
# 6. TRAIN + EVALUATE ALL MODELS
# -----------------------------------
results = []
all_classification_reports = []

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")

    # Train
    model.fit(X_train, y_train)

    # Predict labels
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Predict probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_test_prob = None

    # -------------------------
    # METRICS
    # -------------------------
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='binary', zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # ROC-AUC
    if y_test_prob is not None:
        roc_auc = roc_auc_score(y_test, y_test_prob)
    else:
        roc_auc = np.nan

    # Fit status
    fit_status = check_fit_status(train_acc, test_acc)

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"Specificity    : {specificity:.4f}")
    print(f"ROC-AUC        : {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC        : Not available")
    print(f"Fit Status     : {fit_status}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Save classification report rows
    report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            all_classification_reports.append({
                "Model": model_name,
                "Class": label,
                **metrics
            })

    # Save results
    results.append({
        "Model": model_name,
        "Train Accuracy": round(train_acc, 4),
        "Test Accuracy": round(test_acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "Specificity": round(specificity, 4),
        "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
        "Train-Test Gap": round(train_acc - test_acc, 4),
        "Fit Status": fit_status,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    # -------------------------
    # SAVE CONFUSION MATRIX PLOT
    # -------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Before (0)", "After (1)"],
        yticklabels=["Before (0)", "After (1)"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    safe_model_name = model_name.replace(" ", "_")
    plt.savefig(f"{output_folder}/{safe_model_name}_confusion_matrix.png")
    plt.close()

# -----------------------------------
# 7. CREATE RESULTS DATAFRAME
# -----------------------------------
results_df = pd.DataFrame(results)

# Rank models by F1-score first, then ROC-AUC, then Test Accuracy
results_df = results_df.sort_values(
    by=["F1-score", "ROC-AUC", "Test Accuracy"],
    ascending=False
).reset_index(drop=True)

# Best model highlighting
results_df["Best Model"] = "No"
results_df.loc[0, "Best Model"] = "Yes"

print("\n\nFINAL MODEL COMPARISON TABLE")
print(results_df)

# -----------------------------------
# 8. SAVE TO EXCEL
# -----------------------------------
classification_report_df = pd.DataFrame(all_classification_reports)

excel_path = f"{output_folder}/model_comparison_results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Summary Results", index=False)
    classification_report_df.to_excel(writer, sheet_name="Classification Reports", index=False)

print(f"\nExcel results saved to: {excel_path}")

# Also save CSV if needed
results_df.to_csv(f"{output_folder}/model_comparison_results.csv", index=False)

# -----------------------------------
# 9. VISUALIZATION - TEST ACCURACY
# -----------------------------------
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model", y="Test Accuracy")
plt.xticks(rotation=45)
plt.title("Test Accuracy Comparison Across Models")
plt.tight_layout()
plt.savefig(f"{output_folder}/test_accuracy_comparison.png")
plt.show()

# -----------------------------------
# 10. VISUALIZATION - F1 SCORE
# -----------------------------------
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model", y="F1-score")
plt.xticks(rotation=45)
plt.title("F1-score Comparison Across Models")
plt.tight_layout()
plt.savefig(f"{output_folder}/f1_score_comparison.png")
plt.show()

# -----------------------------------
# 11. VISUALIZATION - ROC-AUC
# -----------------------------------
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model", y="ROC-AUC")
plt.xticks(rotation=45)
plt.title("ROC-AUC Comparison Across Models")
plt.tight_layout()
plt.savefig(f"{output_folder}/roc_auc_comparison.png")
plt.show()

# -----------------------------------
# 12. VISUALIZATION - SPECIFICITY
# -----------------------------------
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model", y="Specificity")
plt.xticks(rotation=45)
plt.title("Specificity Comparison Across Models")
plt.tight_layout()
plt.savefig(f"{output_folder}/specificity_comparison.png")
plt.show()

# -----------------------------------
# 13. FINAL BEST MODEL MESSAGE
# -----------------------------------
best_model_name = results_df.loc[0, "Model"]
best_f1 = results_df.loc[0, "F1-score"]
best_test_acc = results_df.loc[0, "Test Accuracy"]

print("\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"F1-score      : {best_f1}")
print(f"Test Accuracy : {best_test_acc}")
print("="*70)