import numpy as np
import pandas as pd
file_path="Lab Session Data.xlsx"
df=pd.read_excel(file_path,sheet_name="thyroid0387_UCI")
df.replace("?",np.nan,inplace=True)
print(df.head())
print("The number of missing values",df.isna().sum())
number_of_numeric=df.select_dtypes(include=["int64","float64"]).columns
number_of_categories=df.select_dtypes(include=["object"]).columns
print("The number of numeric columns are",number_of_numeric)
print("The number of categorical columns are",number_of_categories)
def has_outliers_iqr(series):
    series=series.dropna()
    if len(series)==0:
        return False
    Q1=series.quantile(0.25)
    Q2=series.quantile(0.75)
    IQR=Q2-Q1
    lower=Q1-1.5*IQR
    upper=Q2+1.5*IQR
    Outliers=series[(series<lower)&(series>upper)]
    return len(Outliers)>0
copied_df=df.copy()
for col in number_of_numeric:
    if copied_df[col].isna().sum()>0:
        if has_outliers_iqr(df[col]):
            median_col=df[col].median()
            copied_df=copied_df.fillna(median_col)
            print(f"The missing values in the {col} column is filled with {median_col}")
        else:
            mean_col=df[col].mean()
            copied_df=copied_df.fillna(mean_col)
            print(f"The column which does not have outliers in {col} column is filled with {mean_col} value")
for col in number_of_categories:
    if copied_df[col].isna().sum()>0:
        mode_col=df[col].mode()[0]
        copied_df=copied_df.fillna(mode_col)
        print(f"The column {col} is filled with the values {mode_col}")
print("Missing values after modification")
print(copied_df.isna().sum())
copied_df.to_excel("Modified_thyroid.xlsx",index=False)