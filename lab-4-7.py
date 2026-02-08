import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
file_path="Lab Session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
data["Class"]=data["Chg%"].apply(lambda x:1 if x>0 else 0)
features=["Price","Open","High","Low"]
Y=data["Class"].values
X=data[features].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.3)
neigh=KNeighborsClassifier()#For creating a model
param_grid={"n_neighbors":list(range(1,21))}
grid=GridSearchCV(neigh,param_grid,cv=5,scoring="accuracy")
grid.fit(X_train,Y_train)
print("The best k values is",grid.best_params_)
best_model=grid.best_estimator_
y_pred=best_model.predict(X_test)
print("The test accuracy is",accuracy_score(Y_test,y_pred))