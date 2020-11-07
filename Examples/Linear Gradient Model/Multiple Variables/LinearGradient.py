import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

# Fetching Data
df = pd.read_csv("homprices.csv")
print('Home Prices Data:- \n {}'.format(df))

print('Data Processing')
median_bedrooms = math.floor(df.bedrooms.median())
print('Median of Bedroom column:- {}'.format(median_bedrooms))
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

print('Preparing Model')
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)

print('Model Co-efficient:- {}'.format(reg.coef_))
print('Model Intercep:- {}'.format(reg.intercept_))

print('Predicting Price for 2000 sq ft., 3 bedrooms, 10 yrs - {}'.format(
    reg.predict([[2000, 3, 10]])[0]))
