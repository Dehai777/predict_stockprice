import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Read file
w1 = pd.read_csv('archive/W1/SBUX.US_W1.csv')

# Set index
w1['datetime'] = pd.to_datetime(w1['datetime'])
w1.set_index('datetime', inplace=True)

# Divide train_data and test_data
train_data = w1['close']['1998':'2019']
test_data = w1['close']['2019':'2024']

# Normalize value
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))

# Set time_stamp
time_stamp = 20

# Train set
x_train, y_train = [], []
for i in range(time_stamp, len(scaled_train_data)):
    x_train.append(scaled_train_data[i - time_stamp:i, 0])
    y_train.append(scaled_train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Test set
x_test, y_test = [], []
for i in range(time_stamp, len(scaled_test_data)):
    x_test.append(scaled_test_data[i - time_stamp:i, 0])
    y_test.append(scaled_test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Create and train the LSTM model
epochs = 75
batch_size = 16
model = Sequential()
model.add(LSTM(units=200, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Predictions
closing_price = model.predict(x_test)

# Inverse normalization
closing_price = scaler.inverse_transform(closing_price)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rms = np.sqrt(np.mean((y_test - closing_price) ** 2))

# Plot the results
plt.figure(figsize=(16, 8))
plt.plot(test_data.index[-len(closing_price):], y_test, color='blue', label='Actual')
plt.plot(test_data.index[-len(closing_price):], closing_price, color='green', label='Predictions')
plt.legend()
plt.title(f'LSTM Predicted Result (Epochs={epochs})\nRMSE: {rms:.2f}')
plt.ylabel('Price')
plt.show()
