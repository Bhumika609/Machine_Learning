import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
file_path="Lab session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
features=["Price","Open","High","Low"]
data["Class"]=data["Chg%"].apply(lambda x:1 if x>0 else 0)
Y=data["Class"].values
X=data[features].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,Y_train)
score=neigh.score(X_test,Y_test)
y_pred=neigh.predict(X_test)
print("the score tat we are getting from this model is",score)
matrix=confusion_matrix(Y_test,y_pred)
print("The confusion matrix is",matrix)
accuarcy=accuracy_score(Y_test,y_pred)
precision=precision_score(Y_test,y_pred)
recall=recall_score(Y_test,y_pred)
f1=f1_score(Y_test,y_pred)
print("The accuarcy score is",accuarcy)
print("the precision score is",precision)
print("The reacll score is",recall)
print("the f1 score is",f1)
train_predict=neigh.predict(X_train)
test_predict=neigh.predict(X_test)
train_accuracy=accuracy_score(Y_train,train_predict)
test_accuarcy=accuracy_score(Y_test,test_predict)
if train_accuracy-test_accuarcy>0.1:
    print("The model is likely to be Overfitiing")
elif test_accuarcy-train_accuracy<0.1:
    print("the model is likely underfitting")
else:
    print("the model is likely to be a regular fit")