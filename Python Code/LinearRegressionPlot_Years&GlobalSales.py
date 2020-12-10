# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:53:01 2020

@author: Kevin
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

path = "C:/Users/Kevin/Documents/GitHub/IntroToAI"

filename_read = os.path.join(path, "vgsales.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

print(df[['Year','Global_Sales']])