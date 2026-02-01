import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
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
distances_numpy=[]
p_val=range(1,11)
for p in range(1,11):
    distance=minkowski_distance(x,y,p)
    distances.append(distance)
    print(f"Minkowski distance for {p} is {distance}")
for p in range(1,11):
    distance_numpy=scipy.spatial.distance.minkowski(x,y,p)
    distances_numpy.append(distance_numpy)
    print(f"Minkowski distance for {p} using scilearn is {distance_numpy}")
comparison = pd.DataFrame({"p value":list(p_val),"Manual Minkowski":distances,"SciPy Minkowski":distances_numpy
})
print(comparison)



