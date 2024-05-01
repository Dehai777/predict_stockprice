import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Read file
df = pd.read_csv('archive/W1/SBUX.US_W1.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Feature and target
x = df['close'].values.reshape(-1, 1)
y = df['close'].values

# Set time_stamp
time_stamp = 20

# Normalize values
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x)

# Train set
x_train_lstm, y_train_lstm = [], []
for i in range(time_stamp, len(x_train_scaled)):
    x_train_lstm.append(x_train_scaled[i - time_stamp:i, 0])
    y_train_lstm.append(x_train_scaled[i, 0])

x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)

# Reshape data for LSTM
x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))

# Create and train the LSTM model
epochs = 50
batch_size = 16
model = Sequential()
model.add(LSTM(units=200, return_sequences=True, input_shape=(x_train_lstm.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=1)

# Predict future one time point
# Extract last 20 data points for prediction
data_to_predict = df['close'][-20:].values.reshape(-1, 1)

# Normalize data
scaled_data_to_predict = scaler.transform(data_to_predict)

# Build feature matrix
x_to_predict = np.array([scaled_data_to_predict[i:i+time_stamp, 0] for i in range(len(scaled_data_to_predict)-time_stamp + 1)])

# Reshape data for LSTM
x_to_predict = np.reshape(x_to_predict, (x_to_predict.shape[0], x_to_predict.shape[1], 1))

# Use model for prediction
predicted_price = model.predict(x_to_predict)

# Inverse normalize predicted price
predicted_price = scaler.inverse_transform(predicted_price)

# Get next week's date
next_week_date = df.index[-1] + pd.Timedelta(days=7)

# Output predicted price
print("Predicted price for the next week:", predicted_price[0][0], "on", next_week_date.strftime('%Y-%m-%d'))
