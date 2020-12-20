# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.metrics import r2_score 

data = pd.read_csv(
    "/Users/Kevin/Documents/Github/IntroToAI/VideoGameSalesWithRating.csv")
print(data.info)
print(data.describe())
print(data.head(100))  # Lists the first Top 100 games from dataset.
print(data.head(100))  # Lists the last Top 100 games from the dataset.

platf_sales = data.groupby('Platform')['Global_Sales'].sum(
).sort_values(ascending=False).reset_index()

# Converts categories columns into numeric values to be used for pearson correlation
cols = ['Platform', 'Genre', 'Publisher']
for col in cols:
    uniqueValue = data[col].value_counts().keys()
    uniques_dict = {}
    num = 0
    for i in uniqueValue:
        uniques_dict[i] = num
        num += 1

    for k, v in uniques_dict.items():
        data.loc[data[col] == k, col] = v

# Compare correlation on Global_Sales against other variables. (Not involving other region sales.)
corr, _ = pearsonr(data.Global_Sales, data.Critic_Score)
print('Global Sales x Critic Score correlation: %.3f' % corr)
corr, _ = pearsonr(data.Global_Sales, data.Year_of_Release)
print('Global Sales x Year_of_Release: %.3f' % corr)
# We disregard this as different platforms released at different times.
# Platform isn't constantly changing
#corr, _ = pearsonr(data.Global_Sales, data.Platform)
#print('Global Sales x Platform: %.3f' % corr)
corr, _ = pearsonr(data.Global_Sales, data.Genre)
print('Global Sales x Genre: %.3f' % corr)
corr, _ = pearsonr(data.Global_Sales, data.Publisher)
print('Global Sales x Publisher: %.3f' % corr)

# Heatmap for pearsons correlation
plt.figure(figsize=(12,12))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Create barchart for global sales against platforms
platf_sales_top10 = platf_sales.iloc[0:10, 0:10]
ax = platf_sales_top10.plot(
    x='Platform', y='Global_Sales', kind='bar', figsize=(23, 5), rot=360)
ax.get_legend().remove()
plt.title('Global Sales by platform', size=14)
plt.xlabel('Platform')
plt.ylabel('Global sales in millions')
plt.show()

# Linear regression plot for Critic Score against Global Sales with best fitted line.
x = data.Global_Sales.values.reshape(-1, 1)
y = data.Critic_Score.values.reshape(-1, 1)

#split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/5, random_state=10)

sns.regplot(x="Critic_Score", y="Global_Sales", data=data, ci=None,
            color="blue", line_kws={"color": "red"}, x_jitter=.02).set(ylim=(0, 17.5))
model = LinearRegression()
model.fit(x,y)
print(model.coef_)


