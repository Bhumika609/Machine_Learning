import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------
# 1. LOAD DATA
# ------------------------------------
df = pd.read_csv("combined_eeg_dataset.csv")

# Remove unnamed columns if any
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Columns in dataset:")
print(df.columns)

# ------------------------------------
# 2. DEFINE FEATURES, LABELS, GROUPS
# ------------------------------------
X = df.drop(columns=["Subject", "Label"])
y = df["Label"]
groups = df["Subject"]

# ------------------------------------
# 3. TRAIN-TEST SPLIT (SUBJECT-WISE)
# ------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# ------------------------------------
# 4. DEFINE MODEL
# ------------------------------------
dt = DecisionTreeClassifier(random_state=42)

# ------------------------------------
# 5. DEFINE HYPERPARAMETER GRID
# ------------------------------------
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 4, 6],
    "class_weight": [None, "balanced"]
}

# ------------------------------------
# 6. CROSS-VALIDATION STRATEGY
# ------------------------------------
group_kfold = GroupKFold(n_splits=5)

# ------------------------------------
# 7. GRID SEARCH
# ------------------------------------
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=group_kfold,
    scoring="f1_macro",   # better than accuracy for imbalanced classes
    n_jobs=-1,
    verbose=2
)

# IMPORTANT: pass groups=groups_train
grid_search.fit(X_train, y_train, groups=groups_train)

# ------------------------------------
# 8. BEST PARAMETERS
# ------------------------------------
print("\nBest Parameters Found:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Score:")
print(grid_search.best_score_)

# ------------------------------------
# 9. BEST MODEL
# ------------------------------------
best_model = grid_search.best_estimator_

# Predict on unseen test subjects
y_pred = best_model.predict(X_test)

# ------------------------------------
# 10. FINAL TEST EVALUATION
# ------------------------------------
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------
# 11. CONFUSION MATRIX PLOT
# ------------------------------------
cm = confusion_matrix(y_test, y_pred)

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
plt.title("Confusion Matrix - Tuned Decision Tree")
plt.tight_layout()
plt.savefig("tuned_decision_tree_confusion_matrix.png")
plt.show()

# ------------------------------------
# 12. FEATURE IMPORTANCE
# ------------------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

plt.figure(figsize=(10, 6))
sns.barplot(
    x=feature_importance["Importance"].head(15),
    y=feature_importance["Feature"].head(15)
)
plt.title("Top 15 Feature Importances - Tuned Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("tuned_decision_tree_feature_importance.png")
plt.show()

# ------------------------------------
# 13. OPTIONAL: TREE VISUALIZATION
# ------------------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=["Before", "After"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Tuned Decision Tree Visualization")
plt.savefig("tuned_decision_tree_visualization.png")
plt.show()