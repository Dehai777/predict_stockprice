from sklearn.linear_model import LinearRegression
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read data
w1 = pd.read_csv('archive/W1/SBUX.US_W1.csv')
# Set index
w1['datetime'] = pd.to_datetime(w1['datetime'])
w1.set_index('datetime', inplace=True)

# Split train and test sets
train_data = w1['close']['1998':'2019']
test_data = w1['close']['2019':'2024']

# Build feature matrix and target vector for training set
X_train = train_data[:-1].values.reshape(-1, 1)  # Use all data except the last day as features
y_train = train_data[1:].values.reshape(-1, 1)   # Shift the target vector by one day

# Build feature matrix and target vector for test set
X_test = test_data[:-1].values.reshape(-1, 1)   # Use all data except the last day as features
y_test = test_data[1:].values.reshape(-1, 1)    # Shift the target vector by one day

# Create linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Calculate root mean squared error
rms = sqrt(mean_squared_error(y_test, predictions))

# Create DataFrame with predicted results
results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()}, index=w1.index[-len(predictions):])

# Print model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
results['Actual'].plot(label='Actual')
results['Predicted'].plot(label='Predicted')
plt.title('Line Regression Predicted (one day) Result\nRMSE: {:.2f}'.format(rms))
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
