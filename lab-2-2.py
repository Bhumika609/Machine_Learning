import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
file_path="Lab Session Data.xlsx"
df=pd.read_excel(file_path,sheet_name="IRCTC Stock Price",usecols=["Date","Month","Day","Price","Open","High","Low","Volume","Chg%"])
print(df.head())
df["Price"]=df["Price"].astype(str).str.replace(",","")
df["Price"]=df["Price"].astype(float)
df["Chg%"]=df["Chg%"].astype(str).str.replace("%","")
df["Chg%"]=df["Chg%"].astype(float)
price_data=df["Price"].values
mean_np=np.mean(price_data)
variance_np=np.var(price_data)
print("the mean using numpy is",mean_np)
print("The variance using numpy is",variance_np)
def mean(arr):
    total=0
    n=len(arr)
    for i in arr:
        total=total+i
    return total/n
def variance(arr):
    n=len(arr)
    total=0
    m=mean(arr)
    for i in arr:
        total=total+(i-m)**2
    return total/n
print("The mean calculated from the own function is",mean(price_data))
print("The variance calculated from teh own function is",variance(price_data))
print("Difference in mean",abs(mean_np-mean(price_data)))
print("Differnece in variance",abs(variance_np-variance(price_data)))
def avg_time(arr,func,run=10):
    times=[]
    for _ in range(run):
       start=time.time()
       func(arr)
       end=time.time()
       times.append(end-start)
    return sum(times)/run
t_np_mean=avg_time(price_data,np.mean)
t_np_variance=avg_time(price_data,np.var)
t_my_mean=avg_time(price_data,mean)
t_my_variance=avg_time(price_data,variance)
print("The time calculated with numpy mean",t_np_mean)
print("The time calculated with numpy variance",t_np_variance)
print("Time form own mean",t_my_mean)
print("Time from own variance",t_my_variance)
wednesday_prices=df[df["Day"]=="Wed"]["Price"]
sample_mean_wed=wednesday_prices.mean()
print("The mean of the wednesday prices",sample_mean_wed)
print("Population of all the mean",mean_np)
print("Difference",abs(sample_mean_wed-mean_np))
april_prices=df[df["Month"]=="Apr"]["Price"]
sample_mean_april=april_prices.mean()
print("The mean of the wednesday prices",sample_mean_april)
print("Population of all the mean",mean_np)
print("Difference",abs(sample_mean_april-mean_np))
loss_days=df[df["Chg%"]<0] #first df is  for the condition and and the second df is for filtering the dataframe
prob_loss=len(loss_days)/len(df)
print("Loss is",prob_loss)
print("Loss(%)",prob_loss*100)
profit_wednesday=df[(df["Day"]=="Wed")&(df["Chg%"]>0)]
prob_profit=len(profit_wednesday)/len(df)
print("The profit on wednesday is",prob_profit*100)
total_wed=df[df["Day"]=="Wed"]
profit_wed_only=total_wed[total_wed["Chg%"]>0]
conditional_prob=len(profit_wed_only)/len(total_wed)
print("P(Profit/Wednesday)",conditional_prob)
day_map={"Mon":1,"Tue":2,"Wed":3,"Thu":4,"Fri":5}
df["DayNum"]=df["Day"].map(day_map)
plt.figure(figsize=(8,5))
plt.scatter(df["DayNum"],df["Chg%"])
plt.xticks([1,2,3,4,5],["Mon","Tue","Wed","Thu","Fri"])
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Scatter Plot: Chg% vs Day of Week")
plt.grid(True)
plt.show()