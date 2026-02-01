import numpy as np
import pandas as pd
import math
data=pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")
print(data.columns)
data["Class"]=data["Chg%"].apply(lambda x:1 if x>=0 else 0)
features=["Price","Open","High","Low"]
class0=data[data["Class"]==0][features].values
class1=data[data["Class"]==1][features].values
print(class0)
print(class1)
def mean_manual(x):
    return sum(x)/len(x)
def var_manual(x):
    n=len(x)
    total=0
    m=mean_manual(x)
    for i in x:
        total=total+(i-m)**2
    return total/n
def std(x):
    variance=var_manual(x)
    return math.sqrt(variance)
def dataset_stats_manual(matrix):
    means=[]
    variances=[]
    stds=[]
    for col in range(matrix.shape[1]):  
        feature_data = matrix[:,col]
        means.append(mean_manual(feature_data))
        variances.append(var_manual(feature_data))
        stds.append(std(feature_data))
    return means,variances,stds
centroid_class0=class0.mean(axis=0)
centroid_class1=class1.mean(axis=0)
print("The centroid of class 0",centroid_class0)
print("The centroid of class 1",centroid_class1)
spread_class_0=np.std(class0,axis=0)
spread_class_1=np.std(class1,axis=0)
print("Intraclass spread of class1",spread_class_0)
print("Intraclass spread of class2",spread_class_1)
interclass_distance=np.linalg.norm(centroid_class1-centroid_class0)
print("The inter class distance is",interclass_distance)
mean0,var0,std0=dataset_stats_manual(class0)
print("Th mean of class0",mean0)
print("The variance if class0",var0)
print("The std of class0",std0)
mean1,var1,std1=dataset_stats_manual(class1)
print("Th mean of class0",mean1)
print("The variance if class0",var1)
print("The std of class0",std1)
