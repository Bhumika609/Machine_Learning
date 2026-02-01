import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
file_path="Lab Session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
data["Class"]=data["Chg%"].apply(lambda x:1 if x>0 else 0)
features=["Price","Open","High","Low"]
Y=data["Class"].values
X=data[features].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
def euclidean_distance(X,Y):
    total=0
    for i in range(len(X)):
        total=total+(X[i]-Y[i])**2
    return math.sqrt(total)
def knearestneighbors(X_train,Y_train,test_point,k):
    distances=[]
    for i in range(len(X_train)):
        dis=euclidean_distance(X_train[i],test_point)
        distances.append((dis,Y_train[i])) #this will ne the distance between the points and the class label of the testing data
    distances.sort(key=lambda x:x[0])
    neighbors=distances[:k]
    count={}
    for dis,class_label in neighbors:
            if class_label in count:
                count[class_label]=count[class_label]+1
            else:
                count[class_label]=1
    return max(count,key=count.get)
def knn_predict(X_train,Y_train,X_test,k):
     predictions=[]
     for test_point in X_test:
          predict=knearestneighbors(X_train,Y_train,test_point,k)
          predictions.append(predict)
     return predictions
knn_library=KNeighborsClassifier(n_neighbors=3)
knn_library.fit(X_train,Y_train)
accuracy_library=knn_library.score(X_test,Y_test)
knn_manually=knn_predict(X_train,Y_train,X_test,3)
correct=0
for i in range(len(Y_test)):
     if knn_manually[i]==Y_test[i]:
          correct=correct+1
accuarcy=correct/len(Y_test)
print("The score for k nearest neighbors using the manual process is",accuarcy)
print("The score from the library is",accuracy_library)