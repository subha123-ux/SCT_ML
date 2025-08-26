import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('House_Price_Prediction/train.csv')
print(f"Loaded dataset: {df}")

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Feature engineering
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
df['Bathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']
print(f"Feature engineered DataFrame: {df[['TotalSF', 'Bathrooms']].head()}")

# Select features and target variable
X = df[['TotalSF', 'BedroomAbvGr', 'Bathrooms']]
y = df['SalePrice']
print(f"X: {X}")
print(f"y: {y}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train: {X_train}")
print(f"X_test: {X_test}")
print(f"y_train: {y_train}")
print(f"y_test: {y_test}")

# Train the linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(f"Predicted values: {y_pred}")

# Evaluate the model
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Display model coefficients
print('Coefficients:')
print(f'  TotalSF coef: {model.coef_[0]:.2f}')
print(f'  Bedrooms coef: {model.coef_[1]:.2f}')
print(f'  Bathrooms coef: {model.coef_[2]:.2f}')

# Display intercept
print(f'Intercept: {model.intercept_:.2f}')

# For individual predictions
TotalSF=int(input("Enter TotalSF: "))
BedroomAbvGr=int(input("Enter number of Bedrooms: "))
Bathrooms=float(input("Enter number of Bathrooms: "))
predicted_price = model.predict([[TotalSF, BedroomAbvGr, Bathrooms]])
print(f'Predicted House Price: {predicted_price[0]:.2f}')

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')  #Set x-axis label
plt.ylabel('Predicted Prices')  #Set y-axis label
plt.title('Actual vs Predicted House Prices')  #Set title
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2) 
plt.show()
