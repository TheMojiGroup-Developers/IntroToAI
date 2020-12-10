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
pd.set_option('display.max_columns', None)

print(df[:400])

print(df.isnull().any())

# Strip non-numeric features from the dataframe
df = df.select_dtypes(include=['int', 'float'])

#print to check that this has worked
print(df[:5]) 

#collect the columns names for non-target features
result = []
for x in df.columns:
    if x != 'Global_Sales':
        result.append(x)
   
X = df[result].values
y = df['Global_Sales'].values

#split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=320, random_state=0)

# build the model
model = LinearRegression()  
model.fit(X_train, y_train)

print(model.coef_)

#calculate the predictions of the linear regression model
y_pred = model.predict(X_test)

#build a new data frame with two columns, the actual values of the test data, 
#and the predictions of the model
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

df_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)   

## RMSE is around 0.4% of the Mean value. This means that it is a relatively good score.
