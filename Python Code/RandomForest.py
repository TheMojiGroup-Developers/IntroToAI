import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
from pprint import pprint

#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# set your path
path_kevin = "C:/Users/Kevin/Documents/GitHub/IntroToAI"
path_saffan = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI/"
# read in the data as csv
filename_read = os.path.join(path_saffan, "VideoGameSalesWithRating.csv")
dataset = pd.read_csv(filename_read)

print(dataset)

x = dataset.iloc[:, 1:2].values  
print(x) 
y = dataset.iloc[:, 2].values 

print(x)
print(y)

# Fitting Random Forest Regression to the dataset 
# import the regressor 
from sklearn.ensemble import RandomForestRegressor 
  
 # create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(x, y)

# print(dataset.shape)
# print(dataset[:5])

# dataset.columns = ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
#                    'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count']

# # Encode the feature values which are strings to integers
# for label in dataset.columns:
#     dataset[label] = LabelEncoder().fit(
#         dataset[label]).transform(dataset[label])

# # Create our X and y data
# result = []
# for x in dataset.columns:
#     if x != 'Name':
#         result.append(x)

# X = dataset[result].values
# y = dataset['Name'].values

# # print(X[:5])

# # Instantiate the model with 10 trees and entropy as splitting criteria
# Random_Forest_model = RandomForestClassifier(
#     n_estimators=10, criterion="entropy")

# # Training/testing split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=7)

# # Train the model
# Random_Forest_model.fit(X_train, y_train)

# # make predictions
# y_pred = Random_Forest_model.predict(X_test)

# # print(y_pred[:5])
# # print(y_test[:5])

# # Calculate accuracy metric
# accuracy = accuracy_score(y_pred, y_test)
# print('The accuracy is: ', accuracy*100, '%')
