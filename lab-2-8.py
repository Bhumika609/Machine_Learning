import pandas as pd
import numpy as np
file_path="Lab Session Data.xlsx"
df=pd.read_excel(file_path,sheet_name="thyroid0387_UCI")
df.replace("?",np.nan,inplace=True)
print(df.head())
numeric_cols=df.select_dtypes(include=["int64","float64"]).columns
print(numeric_cols)
copied_df=df.copy()
for col in numeric_cols:
    copied_df[col]=copied_df[col].fillna(copied_df[col].median())
for col in numeric_cols:
    min=copied_df[col].min()
    max=copied_df[col].max()
    print(f"The minimum of the column {col} is {min} and the maximum of the column {col} is {max}")
df_minmax=copied_df.copy()
for col in numeric_cols:
    min_value=df_minmax[col].min()
    max_value=df_minmax[col].max()
    df_minmax[col]=(df_minmax[col]-min_value)/(max_value-min_value)
df_z_score=copied_df.copy()
for col in numeric_cols:
    mean=df_z_score[col].mean()
    standard_deviation=df_z_score[col].std()
    df_z_score[col]=(df_z_score[col]-mean)/standard_deviation
df_minmax.to_excel("minmax_normalization.xlsx",index=False)
df_z_score.to_excel("zscorenormalization.xlsx",index=False)