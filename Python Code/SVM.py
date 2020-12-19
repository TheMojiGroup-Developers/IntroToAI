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

regressor = SVR(kernel='')
regressor.fit(X, y)

# hello
