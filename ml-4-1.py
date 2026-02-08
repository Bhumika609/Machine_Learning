import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
file_path="Lab Session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
features=["Price","Open","High","Low"]
data["Class"]=data["Chg%"].apply(lambda x:1 if x>0 else 0)
Y=data["Class"].values
X=data[features].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,Y_train)
y_pred_test=neigh.predict(X_test)
y_pred_train=neigh.predict(X_train)
train_accuracy=accuracy_score(Y_train,y_pred_train)
testing_accuracy=accuracy_score(Y_test,y_pred_test)
print("TRAINING RESULTS")
print("THe accoracy score is",accuracy_score(Y_train,y_pred_train))
print("The precision score is",precision_score(Y_train,y_pred_train))
print("The recall score is",recall_score(Y_train,y_pred_train))
print("The f1 score is",f1_score(Y_train,y_pred_train))
print("TESTING RESULTS")
print("TEh accoracy score is",accuracy_score(Y_test,y_pred_test))
print("TEH precision score is",precision_score(Y_test,y_pred_test))
print("The recall score is",recall_score(Y_test,y_pred_test))
print("Teh f1 score is",f1_score(Y_test,y_pred_test))
if train_accuracy>0.85 and (train_accuracy-testing_accuracy)>0.1:
    print("Model is Overfitting")
elif train_accuracy<0.65 and testing_accuracy<0.65:
    print("Model is Underfitting")
else:
    print("Model is Regular Fit")