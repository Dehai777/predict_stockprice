import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold

# Read data
w1 = pd.read_csv('archive/W1/SBUX.US_W1.csv')

# set index
w1['datetime'] = pd.to_datetime(w1['datetime'])
w1.set_index('datetime', inplace=True)

# Split train and test sets
train_data = w1['close']['1998':'2019']
test_data = w1['close']['2019':'2024']

# Build feature matrix and target vector
X_train = np.array([train_data[i:i+8] for i in range(len(train_data)-8)])
y_train = np.array(train_data[8:])

X_test = np.array([test_data[i:i+8] for i in range(len(test_data)-8)])
y_test = np.array(test_data[8:])

# Define the number of folds
k = 5

# Initialize KFold
kf = KFold(n_splits=k)

rms_scores = []

# Iterate over each fold
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    # Get training and validation data for this fold
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Create linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train_fold, y_train_fold)

    # Predict on the validation set
    predictions_fold = model.predict(X_val_fold)

    # Calculate RMSE for this fold
    rms_fold = sqrt(mean_squared_error(y_val_fold, predictions_fold))
    print(f'RMS for fold {fold+1}: {rms_fold}')
    rms_scores.append(rms_fold)

# Calculate the average RMSE across all folds
avg_rms = np.mean(rms_scores)
print(f'Average RMSE across all folds: {avg_rms}')

# Create linear regression model using the entire training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Calculate root mean squared error
rms = sqrt(mean_squared_error(y_test, predictions))
print(f'RMS on test set: {rms}')

# Create DataFrame with predicted results
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=w1.index[-len(predictions):])

# Print model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
results['Actual'].plot(label='Actual')
results['Predicted'].plot(label='Predicted')
plt.title('Linear Regression Predicted by K-Fold Result\nAverage RMSE: {:.2f}'.format(avg_rms))
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
