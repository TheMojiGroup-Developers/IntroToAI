import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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


# Makes the prediction line more smoother
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='magenta')
plt.plot(X_grid, regressor.predict(X_grid), color='green')
##plt.scatter(X, y, color='magenta')
##plt.plot(X, regressor.predict(X), color='green')
plt.title('Support Vector Regression Model')
plt.xlabel('Critic Score')
plt.ylabel('Global Sale')
plt.show()

############################################

x = dataset.Critic_Score.values.reshape(-1, 1)
y = dataset.Global_Sales.values.reshape(-1, 1)

# trains the data set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)
SupportVectorRegModel = SVR()
SupportVectorRegModel.fit(x_train, y_train)

# creates prediction
y_pred = SupportVectorRegModel.predict(x_test)
print(y_pred)

# produce RMSE value
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)
