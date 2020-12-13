# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns

#path = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI"

data = pd.read_csv("C:/Users/Kevin/Documents/GitHub/IntroToAI/vgsales.csv")
#data = pd.read_csv("C:/Users/Kevin/Downloads/vgsales.csv")
print(data.info)
print(data.describe())
print(data.head(100)) #Lists the first Top 100 games from dataset.
print(data.head(100)) #Lists the last Top 100 games from the dataset.

# Plotting Global Sales for Each Platform (Top 100)

# Plotting and Visualising the Data:
plt.scatter(data.Year, data.Global_Sales)
plt.xlabel("Year")
plt.ylabel("Global_Sales")
plt.show()

data.dropna(inplace=True)
data.drop(columns="Rank",inplace=True)
data = data[data["Year"]<=2017.0]
data

df = data.groupby(by  = 'Year').sum()
df.plot.line(figsize=(10,10), grid="on");
plt.ylabel("Sales in million $");

sns.pairplot(df)