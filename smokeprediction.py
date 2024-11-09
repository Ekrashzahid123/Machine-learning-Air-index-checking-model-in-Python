
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('sample_air_quality_data.csv')


print(data.head())


X = data[['Temperature', 'Humidity']]  
y = data['PM2.5']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}') 
print(f'R-squared (R2): {r2}')

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


