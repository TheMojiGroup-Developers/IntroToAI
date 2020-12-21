import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# set your path
path_kevin = "C:/Users/Kevin/Documents/GitHub/IntroToAI"
path_saffan = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI/"
# read in the data as csv
filename_read = os.path.join(path_saffan, "VideoGameSalesWithRating.csv")
dataset = pd.read_csv(filename_read)

X = dataset.iloc[:, 10:11].values.astype(float)
y = dataset.iloc[:, 9:10].values.astype(float)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Not sure what kernel to use. Either "poly", "rbf", or "linear"
regressor = SVR(kernel='rbf')
regressor.fit(X, y.ravel())

plt.scatter(X, y, color='magenta')
plt.plot(X, regressor.predict(X), color='green')
plt.title('Support Vector Regression Model')
plt.xlabel('Critic Score')
plt.ylabel('Global Sales')
plt.show()

