import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===============================
# 1️⃣ Unzip the file
# ===============================
extract_path = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\data_processed\features"

# ===============================
# 2️⃣ Read all Excel files
# ===============================

all_data = []

for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(root, file)
            df = pd.read_excel(file_path)
            all_data.append(df)

# Combine in memory (NOT saving physically)
data = pd.concat(all_data, ignore_index=True)

# ===============================
# 3️⃣ Separate X and y
# ===============================

X = data.iloc[:, :-1].values  # all columns except last
y = data.iloc[:, -1].values   # last column

# ===============================
# 4️⃣ Feature Scaling (IMPORTANT for KNN)
# ===============================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# 5️⃣ Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# 6️⃣ Train KNN (k=5)
# ===============================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy (k=5):", accuracy)


# =====================================================
# 📌 PLOT A: Decision Boundary using PCA (2D)
# =====================================================

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reduced, y, test_size=0.3, random_state=42
)

knn_r = KNeighborsClassifier(n_neighbors=5)
knn_r.fit(X_train_r, y_train_r)

# Mesh grid
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.2),
    np.arange(y_min, y_max, 0.2)
)

Z = knn_r.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.title("KNN Decision Boundary (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# =====================================================
# 📌 PLOT B: Accuracy vs K
# =====================================================

k_values = range(1, 11)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_k = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_k))

plt.figure()
plt.plot(k_values, accuracies)
plt.title("Accuracy vs K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.show()


# =====================================================
# 📌 PLOT C: Confusion Matrix
# =====================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix (k=5)")
plt.show()