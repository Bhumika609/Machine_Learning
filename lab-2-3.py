import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path="Lab Session Data.xlsx"
df=pd.read_excel(file_path,sheet_name="thyroid0387_UCI")
print(df.head())
print("Shape of the datset",df.shape)
print("The column names",df.columns)
print("The dataset info is")
df.info()
df.replace("?",np.nan,inplace=True)
numeric_cols=df.select_dtypes(include=["int64","float64"]).columns
categorical_cols=df.select_dtypes(include=["object"]).columns
print("Numeric colums",numeric_cols)
print("Catgory cols",categorical_cols)
for col in categorical_cols:
    print("Column",col)
    print("Unique values",df[col].unique())
print("Numeric column ranges")
for col in numeric_cols:
    print(f"{col}:min={df[col].min()} max={df[col].max()}")
missing_count=df.isna().sum()
missing_percent=(missing_count/len(df))*100
missing_report=pd.DataFrame({"Missing Count":missing_count,"Missing %":missing_percent})
print("The missing values report is:")
print(missing_report)
for col in numeric_cols:
    mean_val=df[col].mean()
    var_val=df[col].var()
    std_val=df[col].std()
    print(f"{col}:Mean={mean_val:.4f},Variance={var_val:.4f},Std={std_val:.4f}")
def outliers_percentile(series,low=0.01,high=0.99):
    series=series.dropna()
    low_val=series.quantile(low)
    high_val=series.quantile(high)
    outliers=series[(series<low_val)|(series>high_val)]
    return outliers,low_val,high_val
out,low_val,high_val=outliers_percentile(df["age"])
print("Low cutoff",low_val,"High_cutoff",high_val)
print("Count",len(out))