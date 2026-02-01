import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file_path="Lab Session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
x=data["Price"].dropna().values
y=data["Open"].dropna().values
def minkowski_distance(x,y,p):
    total=0
    for i in range(len(x)):
        total=total+abs(x[i]-y[i])**p
    return total**(1/p)
distances=[]
p_val=range(1,11)
for p in range(1,11):
    distance=minkowski_distance(x,y,p)
    distances.append(distance)
    print(f"Minkowski distance for {p} is {distance}")
plt.plot(p_val,distances)
plt.xlabel("p_values")
plt.ylabel("minkowski distances")
plt.title("The plot of minkowski Distances")
plt.show()


