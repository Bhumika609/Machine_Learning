import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
file_path="Lab Session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
data["Class"]=data["Chg%"].apply(lambda x:1 if x>0 else 0)
features=["Price","Open","Low","High"]
X=data[features].values
Y=data["Class"].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,Y_train)