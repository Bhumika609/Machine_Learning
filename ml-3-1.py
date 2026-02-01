import pandas as pd
import numpy as np
import math
data=pd.read_excel("vectors.csv.xlsx")
A=data["A"].values
B=data["B"].values
print("THe first vector A",A)
print("The second vector B",B)
def dot_product(A,B):
    result=0
    for i in range(len(A)):
        result+=A[i]*B[i]
    return result
def euclidean_norm(A):
    sum_sq=0
    for i in range(len(A)):
        sum_sq=sum_sq+A[i]**2
    return math.sqrt(sum_sq)
print("THe dot product manullay is",dot_product(A,B))
print("The euclidean norm for A manually",euclidean_norm(A))
print("The euclidead norm for B manually",euclidean_norm(B))
print("The dot product with numpy",np.dot(A,B))
print("The norm of A form numpy is",np.linalg.norm(A))
print("The norm of B using numpy is",np.linalg.norm(B))
