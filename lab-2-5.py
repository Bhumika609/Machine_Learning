import pandas as pd
import numpy as np
file_path="Lab Session Data.xlsx"
df=pd.read_excel(file_path,sheet_name="thyroid0387_UCI")
df.replace("?",np.nan,inplace=True)
df=df.replace({"t":1,"f":0})
df_encoded=pd.get_dummies(df,drop_first=False)
df_encoded=df_encoded.apply(pd.to_numeric)
df_encoded=df_encoded.fillna(df_encoded.mean())
A=df_encoded.iloc[0].values
B=df_encoded.iloc[1].values
dot_product=np.dot(A,B)
norm_A=np.linalg.norm(A)
norm_B=np.linalg.norm(B)
cosine_similarity=dot_product/(norm_A*norm_B)
print("Cosine similarity",cosine_similarity)