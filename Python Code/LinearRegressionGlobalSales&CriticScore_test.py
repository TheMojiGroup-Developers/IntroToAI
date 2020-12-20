# Linear Regression & R2 Regression
# Author - Saffan Ahmed

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
sns.set()


# set your path
path_kevin = "C:/Users/Kevin/Documents/GitHub/IntroToAI"
path_saffan = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI/"
# read in the data as csv
filename_read = os.path.join(path_saffan, "LinearRegressionGlobalSales&CriticScore.csv")
dataset = pd.read_csv(filename_read)

plt.title('Linear Regression: Global Sales & Critic Score') 
plt.xlabel('Critic Score') 
plt.ylabel('Global Sales') 
plt.scatter(dataset['Critic_Score'], dataset['Global_Sales'])

X = np.array(dataset['Critic_Score']).reshape(-1, 1)
y = dataset['Global_Sales']
rf = LinearRegression()
rf.fit(X, y)
y_pred = rf.predict(X)


plt.scatter(dataset['Critic_Score'], dataset['Global_Sales'])
plt.plot(X, y_pred, color='red')

def r2_score_from_scratch(ys_orig, ys_line):
    y_mean_line = [ys_orig.mean() for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))
r_squared = r2_score_from_scratch(y, y_pred)
print(r_squared)

r2_score(y, y_pred)