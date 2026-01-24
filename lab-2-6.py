import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path="Lab Session Data.xlsx"
df=pd.read_excel(file_path,sheet_name="thyroid0387_UCI")
df.replace("?",np.nan,inpace="True")
print(df.head())
df20=df.iloc[:20].copy()
binary_cols=[]
for cols in df20.columns:
    unique_values=df20[cols].dropna().unique()
    if set(unique_values).issubset({"t","f"}):
        binary_cols.append(unique_values)
print("Total number of binary columna is",len(binary_cols))
bin_values=df20[binary_cols].map({"t":1,"f":0})
bin_values=bin_values.fillna(0)
print(bin_values.head())
def jc(a,b):
    f00=((a==0)&(b==1)).sum()
    f01=((a==0)&(b==1)).sum()
    f10=((a==1)&(a==0)).sum()
    f11=((a==1)&(a==1)).sum()
    sum=f01+f10+f11
    return f11/sum
def smc(a,b):
    f00=((a==0)&(b==1)).sum()
    f01=((a==0)&(b==1)).sum()
    f10=((a==1)&(a==0)).sum()
    f11=((a==1)&(a==1)).sum()
    sum=f00+f01+f10+f11
    sum_u=f00+f11
    return sum_u/sum
def cos_similarity(a,b):
    dot_product=np.dot(a,b)
    mag_a=np.linalg.norm(a)
    mag_b=np.linalg.norm(b)
    return dot_product/mag_a*mag_b
n=len(bin_values)
jc_matrix=np.zeros((n,n))
smc_matrix=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        a=bin_values.iloc[i].values()
        b=bin_values.iloc[j].values()
        jc_matrix[i,j]=jc(a,b)
        smc_matrix[i,j]=smc(a,b)
df20_full=df20.copy()
df20_full=df20.replace({"t":1,"f":0})
df20_full_encoded=pd.get_dummies(df20_full)
df20_full_encoded=df20_full_encoded.apply(pd.get_numeric())
df20_full_encoded=df20_full_encoded.fillna(df20_full_encoded.mean())
cos_matrix=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        a=df20_full_encoded.iloc[i].values()
        b=df20_full_encoded.iloc[j].values()
        cos_matrix=cos_similarity(a,b)
plt.figure(figsize=(10, 7))
sns.heatmap(jc_matrix,annot=True,fmt=".2f")
plt.title("Heatmap-Jaccard Coefficient (JC)")
plt.show()
plt.figure(figsize=(10, 7))
sns.heatmap(smc_matrix,annot=True,fmt=".2f")
plt.title("Heatmap-SMC Coefficient (JC)")
plt.show()
plt.figure(figsize=(10, 7))
sns.heatmap(cos_matrix,annot=True,fmt=".2f")
plt.title("Heatmap-Cosine Coefficient (JC)")
plt.show()

