import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('VideoGameSalesWithRating.csv')
X = dataset.iloc[:, 10:11].values.astype(float)
y = dataset.iloc[:, 9:10].values.astype(float)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Not sure what kernel to use. Either "poly", "rbf", or "linear"
regressor = SVR(kernel='linear')
regressor.fit(X, y.ravel())

plt.scatter(X, y, color='magenta')
plt.plot(X, regressor.predict(X), color='green')
plt.title('Support Vector Regression Model')
plt.xlabel('Critic Score')
plt.ylabel('Global Sale')
plt.show()
