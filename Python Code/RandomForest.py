import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
from pprint import pprint

#from sklearn.ensemble import RandomForestClassifier
# Fitting Random Forest Regression to the dataset 
# import the regressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

# set your path
path_kevin = "C:/Users/Kevin/Documents/GitHub/IntroToAI"
#path_saffan = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI/"
# read in the data as csv
filename_read = os.path.join(path_kevin, "RandomForestGlobalSales&CriticScore.csv")
dataset = pd.read_csv(filename_read)

print(dataset)

x = dataset.iloc[:, 2:3].values  
y = dataset.iloc[:, 3].values 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(X_train, y_train)

# predicting a new result 
Y_pred = regressor.predict(X_test)  

# Visualising the Random Forest Regression results 
  
# arange for creating a range of values 
# from min value of x to max  
# value of x with a difference of 0.01  
# between two consecutive values 
X_grid = np.arange(min(x), max(x), 0.01)  
  
# reshape for reshaping the data into a len(X_grid)*1 array,  
# i.e. to make a column out of the X_grid value                   
X_grid = X_grid.reshape((len(X_grid), 1)) 
  
# Scatter plot for original data 
plt.scatter(x, y, color = 'blue')   
  
# plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid),  
         color = 'red')  
plt.title('Random Forest Regression: Global Sales & Critic Score') 
plt.xlabel('Critic Score') 
plt.ylabel('Global Sales') 
plt.show()

# Evaluating the Random Forest Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))

