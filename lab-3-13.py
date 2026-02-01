import pandas as pd
import numpy as np
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
def manual_confusion_matrix(y_true,y_pred):
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(len(y_true)):
        if y_true[i]==1 and y_pred[i]==1:
            tp=tp+1
        elif y_true[i]==0 and y_pred[i]==1:
            fp=fp+1
        elif y_true[i]==1 and y_pred[i]==0:
            fn=fn+1
        else:
            tn=tn+1
    return tp,fp,fn,tn
def manual_accuracy(tp,tn,fp,fn):
    return (tp+tn)/(tp+tn+fp+fn)
def precision_manual(tp,fp):
    return tp/(tp+fp)
def recall_manual(tp,fn):
    return tp/(tp+fn)
def f1_score(pr,re):
    return (2*pr*re)/(pr+re)
y_pred=neigh.predict(X_test)
tp,fp,fn,tn=manual_confusion_matrix(Y_test,y_pred)
a=manual_accuracy(tp,tn,fp,fn)
p=precision_manual(tp,fp)
r=recall_manual(tp,fn)
f1=f1_score(p,r)
print("The accuracy is",a)
print("The precision is",p)
print("The recall is",r)
print("the f1 score is",f1)
matrix=np.array([[tp,fp],[fn,tn]])
print("the confusion matrix is",matrix)