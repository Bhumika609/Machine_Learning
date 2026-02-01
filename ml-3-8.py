import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
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
print("the score tat we are getting from this model is",score)