import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
file_path="Lab Session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
data["Class"]=data["Chg%"].apply(lambda x:1 if x>=0 else 0)
features=["Price","Open","High","Low"]
Y=data["Class"].values
X=data[features].values#as features is not a column in your dataset
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
print("X_train Shape",X_train.shape)
print("Y_train shape",Y_train.shape)
print("X_test shape",X_test.shape)
print("Y_test shape",Y_test.shape)