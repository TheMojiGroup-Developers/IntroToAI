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

med = df['Year'].median()
df['Year'] = df['Year'].fillna(med)
#df = df.dropna() #you can also simply drop NA values

print(df.isnull().any())

print(df[:400])

print(df.isnull().any())

