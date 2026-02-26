import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error
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
y_test_predict=reg.predict(X_test)
train_mse=mean_squared_error(Y_train,y_train_predict)
test_mse=mean_squared_error(Y_test,y_test_predict)
train_rmse=np.sqrt(train_mse)
test_rmse=np.sqrt(test_mse)
train_mape=mean_absolute_percentage_error(Y_train,y_train_predict)
test_mape=mean_absolute_percentage_error(Y_test,y_test_predict)
train_r2_score=r2_score(Y_train,y_train_predict)
test_r2_score=r2_score(Y_test,y_test_predict)
print(f"The mean squred error for train is {train_mse} and for test data is {test_mse}")
print(f"The root mean squred error for train is {train_rmse} and for test data is {test_rmse}")
print(f"The mean absolute percentage error for train is {train_mape} and for test data is {test_mape}")
print(f"The mr2 score for train is {train_r2_score} and for test data is {test_r2_score}")