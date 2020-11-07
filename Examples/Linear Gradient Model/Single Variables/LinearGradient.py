import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Fetching Data
df = pd.read_csv("homprices.csv")
print('Home Prices Data:- \n {}'.format(df))

# Creating XY axis chart
plt.xlabel('area(sqr ft.)')
plt.ylabel('US($)')
plt.scatter(df.areas, df.prices, color='red', marker='+')

print('Preparing Model')
reg = linear_model.LinearRegression()
reg.fit(df[['areas']], df.prices)

print('Model Co-efficient:- {}'.format(reg.coef_))
print('Model Intercep:- {}'.format(reg.intercept_))

print('Predicting Price for 2000 sq ft. {}'.format(reg.predict([[2000]])[0]))

print('Created XY Axis Chart and plotting predicting line for current Data')
plt.plot(df.areas, reg.predict(df[['areas']]), color='blue')
