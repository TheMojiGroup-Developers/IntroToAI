# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#path = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI"

data = pd.read_csv(
    "/Users/Kevin/Documents/Github/IntroToAI/VideoGameSalesWithRating.csv")
print(data.info)
print(data.describe())
print(data.head(100))  # Lists the first Top 100 games from dataset.
print(data.head(100))  # Lists the last Top 100 games from the dataset.

#Linear regression plot for Critic Score against Global Sales with best fitted line.

x=data.NA_Sales.values.reshape(-1,1)
y=data.Critic_Score.values.reshape(-1,1)
sns.regplot(x="Critic_Score", y="Global_Sales", data=data, ci=None, color="blue", line_kws={"color": "red"}, x_jitter=.02).set(ylim=(0, 17.5))
model=LinearRegression()
model.fit(x,y)
print(model.coef_)
model.intercept_
x_input=[[5]]
y_predict=model.predict(x_input)
y_predict

