from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
import matplotlib.pyplot as plt
extract_path = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\data_processed\features"
all_data=[]
for root,dirs,files in os.walk(extract_path):
    for file in files:
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path=os.path.join(root, file)
            df=pd.read_excel(file_path)
            all_data.append(df)
data=pd.concat(all_data,ignore_index=True)
X=data.iloc[:,:-1].values
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
k_values=range(2,20)
distortions=[]
for i in k_values:
    k_means=KMeans(n_clusters=i,random_state=0,n_init="auto")
    k_means.fit(X_scaled)
    distortions.append(k_means.inertia_)
print(distortions)
plt.figure()
plt.plot(k_values,distortions,marker="o")
plt.xlabel(k_values)
plt.ylabel(distortions)
plt.title("Elbow method to find out the optimal point for a clustering")
plt.show()