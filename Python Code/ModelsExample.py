# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:50:21 2020

@author: Kevin
"""

import os
from sklearn.model_selection import KFold
from sklearn import datasets
import pandas as pd
import numpy as np

path = path = "/Users/saffanahmed/Documents/IntroToAI/IntroToAI" #relative or absolute path to data

#***********************Training/Validation Split***************************

filename_read = os.path.join(path, "vgsales.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])
df = df.reindex(np.random.permutation(df.index)) # Usually a good idea to shuffle
mask = np.random.rand(len(df)) < 0.8
trainDF = pd.DataFrame(df[mask])
validationDF = pd.DataFrame(df[~mask])

print(f"Training DF: {len(trainDF)}")
print(f"Validation DF: {len(validationDF)}")

#***********************      K-Fold split       ***************************
kf = KFold(5)

fold = 1
for train_index, validate_index in kf.split(df):
    trainDF = pd.DataFrame(df.iloc[train_index, :])
    validateDF = pd.DataFrame(df.iloc[validate_index])
    print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    fold += 1
    
    
# ********************* Representing data as table *********************   
iris = datasets.load_iris() # Replace with own dataset.

#convert the data to a dataframe for display
df = pd.DataFrame(data=np.c_[iris['data'],iris['target']], 
                  columns=iris['feature_names'] + ['target'])

#display data structure and first five rowsHere each row of the data refers to a single observed flower, and the number of rows is the total number of flowers in the dataset. In general, we will refer to the rows of the matrix as **samples**, and the number of rows as *n_samples*.
df.head()