import pandas as pd
import numpy as np
file_path="Lab Session Data.xlsx"
df=pd.read_excel(file_path,sheet_name="thyroid0387_UCI")
df.replace("?",np.nan,inplace=True)
print(df.head(2))
v1=df.iloc[0]
v2=df.iloc[1]
binary_cols=[]
for cols in df.columns:
    unique_values=df[cols].dropna().unique()
    if set(unique_values).issubset({"t","f"}):
        binary_cols.append(cols)
print("Binary columns are",binary_cols)
print("Total binary columns",len(binary_cols))
bin_v1=v1[binary_cols].map({"t":1,"f":0})
bin_v2=v2[binary_cols].map({"t":1,"f":0})
print("Binary vector 1:",bin_v1)
print("Binary vector 2:",bin_v2)
f11=((bin_v1==1)&(bin_v2==1)).sum()
f01=((bin_v1==0)&(bin_v2==1)).sum()
f10=((bin_v1==1)&(bin_v2==0)).sum()
f00=((bin_v1==0)&(bin_v2==0)).sum()
print("f11",f11)
print("f10",f10)
print("f01",f01)
print("f00",f00)
jc=f11/(f11+f01+f10)
print("Jaccard coefficient is",jc)
smc=(f11+f00)/(f11+f10+f01+f00)
print("Simple matching coefficient is",smc)
if smc>jc:
    print("SMC is usually higher as it counts 0-0 also")
else:
    print("JC may be higher when 1-1 matches dominate")
