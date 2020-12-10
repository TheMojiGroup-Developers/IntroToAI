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

## check for null data
print(df.isnull().any())# Strip non-numeric features from the dataframe
df = df.select_dtypes(include=['int', 'char'])
#collect the columns names for non-target features
result = []
for x in df.columns:
    if x != 'NA_Sales':
        result.append(x)
   
X = df[result].values
y = df['NA_Sales'].values


#print to check that this has worked
print(df[:400]) 


