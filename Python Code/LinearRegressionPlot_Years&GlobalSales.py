# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
import scipy.stats


from sklearn import metrics

#path = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI"
#path = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI/VideoGameSalesWithRating.csv"

data = pd.read_csv(
    "/Users/saffanahmed/Documents/IntroToAI/IntroToAI/VideoGameSalesWithRating.csv")
print(data.info)
print(data.describe())
print(data.head(100))  # Lists the first Top 100 games from dataset.
print(data.head(100))  # Lists the last Top 100 games from the dataset.

# Plotting Global Sales for Each Platform (Top 100)

# Plotting and Visualising the  Data:
plt.scatter(data.Year_of_Release, data.Global_Sales)
plt.xlabel("Year_of_Release")
plt.ylabel("Global_Sales")
plt.show()

# Plotting data with sub-plot:
data_plot = data.loc[:, ["Year_of_Release", "Global_Sales"]]
data_plot.plot()
#filename_read = os.path.join(path, "vgsales.csv")
#df = pd.read_csv(filename_read, na_values=['NA', '?'])
# print(df[['Year','Global_Sales']])

# Matrix
matrix = data.corr()
plt.figure(figsize=(8, 6))
# plot heat map
g = sns.heatmap(matrix, annot=True, cmap="YlGn_r")
# Value closest to 1 shows a strong correlation.

sns.pairplot(data)

pearson_coef, p_value = stats.pearsonr(data["Critic_Score"], data["Global_Sales"])
print("Pearson Correlation Co-efficient: ", pearson_coef, " and P Value of:", p_value)

# Global Sales / Critic Score Scatter Plot
df = pd.read_csv("/Users/saffanahmed/Documents/IntroToAI/IntroToAI/VideoGameSalesWithRating.csv")
df.head()
x=df.Global_Sales.values.reshape(-1,1)
y=df.Critic_Score.values.reshape(-1,1)
sns.lmplot(y='Global_Sales',x='Critic_Score',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[5]]
y_predict=model.predict(x_input)
y_predict

# NA_Sales / Critic Score Scatter Plot
df = pd.read_csv("/Users/saffanahmed/Documents/IntroToAI/IntroToAI/VideoGameSalesWithRating.csv")
df.head()
x=df.NA_Sales.values.reshape(-1,1)
y=df.Critic_Score.values.reshape(-1,1)
sns.lmplot(y='NA_Sales',x='Critic_Score',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[5]]
y_predict=model.predict(x_input)
y_predict

# EU_Sales / Critic Score Scatter Plot
df = pd.read_csv("/Users/saffanahmed/Documents/IntroToAI/IntroToAI/VideoGameSalesWithRating.csv")
df.head()
x=df.EU_Sales.values.reshape(-1,1)
y=df.Critic_Score.values.reshape(-1,1)
sns.lmplot(y='EU_Sales',x='Critic_Score',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[5]]
y_predict=model.predict(x_input)
y_predict

# JP_Sales / Critic Score Scatter Plot
df = pd.read_csv("/Users/saffanahmed/Documents/IntroToAI/IntroToAI/VideoGameSalesWithRating.csv")
df.head()
x=df.JP_Sales.values.reshape(-1,1)
y=df.Critic_Score.values.reshape(-1,1)
sns.lmplot(y='JP_Sales',x='Critic_Score',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[5]]
y_predict=model.predict(x_input)
y_predict

# Other_Sales / Critic Score Scatter Plot
df = pd.read_csv("/Users/saffanahmed/Documents/IntroToAI/IntroToAI/VideoGameSalesWithRating.csv")
df.head()
x=df.Other_Sales.values.reshape(-1,1)
y=df.Critic_Score.values.reshape(-1,1)
sns.lmplot(y='Other_Sales',x='Critic_Score',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[5]]
y_predict=model.predict(x_input)
y_predict



