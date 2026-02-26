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
k_values=range(2,11)
sil_scores=[]
cal_score=[]
dav_score=[]
for i in k_values:
    k_means=KMeans(n_clusters=i,random_state=0,n_init="auto")
    k_means.fit(X_scaled)
    cluster_labels=k_means.labels_
    cluster_centers=k_means.cluster_centers_
    sil_scores.append(silhouette_score(X_scaled,cluster_labels))
    cal_score.append(calinski_harabasz_score(X_scaled,cluster_labels))
    dav_score.append(davies_bouldin_score(X_scaled,cluster_labels))
print("The silhoutte score is",sil_scores)
print("The calinski harabasz score is",cal_score)
print("The davies bouldin score is",dav_score)
plt.figure()
plt.plot(k_values,sil_scores)
plt.xlabel(k_values)
plt.ylabel(sil_scores)
plt.title("Silhoutte scores v/s k")
plt.show()
plt.figure()
plt.plot(k_values,cal_score)
plt.xlabel(k_values)
plt.ylabel(cal_score)
plt.title("Calinski-Harabasz scores v/s k")
plt.show()
plt.figure()
plt.plot(k_values,dav_score)
plt.xlabel(k_values)
plt.ylabel(dav_score)
plt.title("Davies-Bouldin scores v/s k")
plt.show()