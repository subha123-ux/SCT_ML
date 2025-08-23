import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('House_Price_Prediction/train.csv')
print(df)

# Data preprocessing
df = df.isnull()  # Check for missing values
df = df.dropna()  # Drop rows with missing values

# Display basic information about the dataset
print(df.info())
print(df.describe())

