import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file_path="Lab Session Data.xlsx"
data=pd.read_excel(file_path,sheet_name="IRCTC Stock Price")
features=data["Price"].dropna().values
print("The mean of the values",np.mean(features))
print("The variance of the values",np.var(features))
hist_values,bin_boundary=np.histogram(features,bins=10)
print("Histogram values",hist_values)
print("The bin boundary",bin_boundary)
plt.hist(features,bins=10)
plt.xlabel("prices")
plt.ylabel("frequency")
plt.title("Frequency plot of prices")
plt.show()