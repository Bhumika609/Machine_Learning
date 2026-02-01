import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
file_path="Lab Session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
data["Class"]=data["Chg%"].apply(lambda x:1 if x>0 else 0)
features=["Price","Open","Low","High"]
X=data[features].values
Y=data["Class"].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,Y_train)
prediction=neigh.predict(X_test)
accuracy=accuracy_score(Y_test,prediction)
##Matrix Inversion Method
X_train_bias=np.c_[np.ones(X_train.shape[0]),X_train]
X_test_bias=np.c_[np.ones(X_test.shape[0]),X_test]
Y_train_col=Y_train.reshape(-1,1)
W = np.linalg.inv(X_train_bias.T@X_train_bias)@X_train_bias.T@Y_train_col
matrix_pred=X_test_bias@W
matrix_pred_class=(matrix_pred>0.5).astype(int).flatten()
matrix_accuracy=accuracy_score(Y_test, matrix_pred_class)
print("kNN Accuracy",accuracy)
print("Matrix Inversion Accuracy",matrix_accuracy)
if accuracy>matrix_accuracy:
    print("kNN performs better so it might be  Data likely non-linear")
elif matrix_accuracy>accuracy:
    print("Matrix inversion performs better so Data likely linear")
else:
    print("Both models perform similarly")
