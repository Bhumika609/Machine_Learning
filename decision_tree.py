import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load combined dataset
df = pd.read_csv("combined_eeg_dataset.csv")

# Remove unnamed columns if any
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Columns in dataset:")
print(df.columns)

# Define features, labels, and groups
X = df.drop(columns=["Subject", "label"])
y = df["label"]
groups = df["Subject"]

# Subject-wise train-test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Create Decision Tree model
#model = DecisionTreeClassifier(
 #   max_depth=5,
  #  min_samples_split=4,
   # min_samples_leaf=2,
    #random_state=42
#) #this is for the balancing of the classes ars the 1 are and 0 are not in equalt the precision and recall shpuold not ignore the 1 that is reason we are adding the class balance
#model=DecisionTreeClassifier(
 #   max_depth=5,
  #  min_samples_split=4,
   # min_samples_leaf=2,
    #class_weight="balanced",
    #random_state=42
#)
#this is for increasing the dept of the tree ro chec if the acoracy or porecsion or recakl is increasing ir not
model=DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 1. CONFUSION MATRIX HEATMAP
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Before (0)", "After (1)"],
            yticklabels=["Before (0)", "After (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.show()

# -------------------------------
# 2. FEATURE IMPORTANCE
# -------------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    x=feature_importance["Importance"].head(15),
    y=feature_importance["Feature"].head(15)
)
plt.title("Top 15 Feature Importances - Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -------------------------------
# 3. DECISION TREE VISUALIZATION
# -------------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Before", "After"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Visualization")
plt.show()