# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#path = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI"

data = pd.read_csv("/Users/saffanahmed/Documents/IntroToAI/IntroToAI/vgsales.csv")
print(data.info)
print(data.describe())
print(data.head(100)) #Lists the first Top 100 games from dataset.
print(data.head(100)) #Lists the last Top 100 games from the dataset.

# Plotting Global Sales for Each Platform (Top 100)

# Plotting and Visualising the  Data:
plt.scatter(data.Name, data.Global_Sales)
plt.xlabel("Name")
plt.ylabel("Global_Sales")
plt.show()

# Plotting data with sub-plot:
data_plot = data.loc[:,["Name","Global_Sales"]]
data_plot.plot()
#filename_read = os.path.join(path, "vgsales.csv")
#df = pd.read_csv(filename_read, na_values=['NA', '?'])
#print(df[['Year','Global_Sales']])

