import pandas as pd
import numpy as np
file_path="Lab Session data.xlsx"
df=pd.read_excel(file_path,sheet_name="Purchase data")
print(df.head())
numeric_df=df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)","Payment (Rs)"]]
print(numeric_df)
square_matrix_1=numeric_df.iloc[0:4,0:4].to_numpy()
print(square_matrix_1.shape)
print("The first square matrix is",square_matrix_1)
square_matrix_2=numeric_df.iloc[5:9,0:4].to_numpy()
print(square_matrix_2.shape)
print("The second square matrix is",square_matrix_2)
pd.DataFrame(square_matrix_1,columns=numeric_df.columns).to_excel("square_matrix_1.xlsx",index=False)
pd.DataFrame(square_matrix_2,columns=numeric_df.columns).to_excel("square_matrix_2.xlsx",index=False)
