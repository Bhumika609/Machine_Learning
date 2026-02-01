import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
file_path="Lab session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
features=["Price","Open","High","Low"]
data["Class"]=data["Chg%"].apply(lambda x:1 if x>0 else 0)
Y=data["Class"].values
X=data[features].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
neigh1=KNeighborsClassifier(n_neighbors=3)
neigh1.fit(X_train,Y_train)
score1=neigh1.score(X_test,Y_test)
neigh2=KNeighborsClassifier(n_neighbors=1)
neigh2.fit(X_train,Y_train)
score2=neigh2.score(X_test,Y_test)
comparision_dataframe=pd.DataFrame({"score with 3 neigthbors":[score1],"score with 1 neightbor":[score2]})#if the column value has multiple numbers then list wrapping is not needed otherwise list wrapping is needed
print(comparision_dataframe)
comp=pd.DataFrame({"Neighbors":[3,1],"Score":[score1,score2]})
print(comp)
k_val=range(1,12)
accuracy_vals=[]
for k in k_val:
    n=KNeighborsClassifier(n_neighbors=k)
    n.fit(X_train,Y_train)
    score=n.score(X_test,Y_test)
    accuracy_vals.append(score)
plt.plot(k_val,accuracy_vals)
plt.xlabel("k_values")
plt.ylabel("accuracy_score")
plt.show()