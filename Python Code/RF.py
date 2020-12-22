# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:07:53 2020

@author: Kevin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
from pprint import pprint

from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# set your path
path_kevin = "C:/Users/Kevin/Documents/GitHub/IntroToAI"
#path_saffan = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI/"
# read in the data as csv
filename_read = os.path.join(path_kevin, "RandomForestGlobalSales&CriticScore.csv")
dataset = pd.read_csv(filename_read)

x = dataset.iloc[:, 2:3].values  
print(x) 
y = dataset.iloc[:, 3].values 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

