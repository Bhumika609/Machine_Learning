import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
extract_path = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\data_processed\features"
all_data=[]
for root,dirs,files in os.walk(extract_path):
    for file in files:
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path=os.path.join(root, file)
            df=pd.read_excel(file_path)
            all_data.append(df)
data=pd.concat(all_data,ignore_index=True)
X=data.iloc[:,0].values.reshape(-1,1) #x must be 2d for linear regression so we need to reshape it again
y=data.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=42)
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_train_predict=reg.predict(X_train)
#model Equation
print("slope",reg.coef_[0])
print("Intercept",reg.intercept_)