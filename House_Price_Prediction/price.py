import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('House_Price_Prediction/train.csv')
print(df)

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Feature engineering
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
df['Bathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']
print(df)

# Select features and target variable
X = df[['TotalSF', 'BedroomAbvGr', 'Bathrooms']]
y = df['SalePrice']
print(X)
print(y)