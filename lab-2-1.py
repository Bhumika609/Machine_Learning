import pandas as pd
import numpy as np
file_path="Lab Session Data.xlsx"
df=pd.read_excel(file_path,sheet_name="Purchase data",usecols=["Candies (#)","Mangoes (Kg)","Milk Packets (#)","Payment (Rs)"])
X=df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]].to_numpy()
Y=df[["Payment (Rs)"]].to_numpy()
print(X)
print(Y)
print("The shape of the matrix X is",X.shape)
print("The shape of y is",Y.shape)
rank_X=np.linalg.matrix_rank(X)
c=np.linalg.pinv(X)@Y
print("The rank of the feature vector matrix is",rank_X)
print("The values for c:",c)
print("The cost of one candy is",c[0][0])
print("The cost of one mango is",c[1][0])
print("the cost of one milk packet is",c[2][0])
df["Payment (Rs)"]=pd.to_numeric(df["Payment (Rs)"])
status=[]
for payment in df["Payment (Rs)"]:
    if payment>200:
        status.append("Rich")
    else:
        status.append("Poor")
df["Status"]=status
output_file="output.xlsx"
df.to_excel(output_file,sheet_name="Purchase data",index=False)