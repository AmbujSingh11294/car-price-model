# Car Price Prediction Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = '/content/sample_data/CarPrice.csv'
car_data = pd.read_csv(file_path)

# Identifying categorical variables
categorical_vars = car_data.select_dtypes(include=['object']).columns

# Applying one-hot encoding to categorical variables
car_data_encoded = pd.get_dummies(car_data, columns=categorical_vars, drop_first=True)

# Splitting the data into training and testing sets
X = car_data_encoded.drop('price', axis=1)
y = car_data_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the prices
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Coefficients of the model
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
