import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow logging to only display errors (3)


def train_model():
    time_stamp = 20
    try:
        # Retrieve inputs from UI
        filename = file_entry.get()
        epochs = int(epochs_entry.get())
        complexity = complexity_var.get()

        # Read file
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        # Feature and target
        x = df['close'].values.reshape(-1, 1)
        y = df['close'].values

        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        # Set model complexity
        units_1 = 100
        units_2 = 50
        if complexity == "Simple":
            units_1 = 50
            units_2 = 25
        elif complexity == "Moderate":
            units_1 = 100
            units_2 = 50
        elif complexity == "Complex":
            units_1 = 200
            units_2 = 100

        # Normalize values
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Train set
        x_train_lstm, y_train_lstm = [], []
        for i in range(time_stamp, len(x_train_scaled)):
            x_train_lstm.append(x_train_scaled[i - time_stamp:i, 0])
            y_train_lstm.append(x_train_scaled[i, 0])

        x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)

        # Test set
        x_test_lstm, y_test_lstm = [], []
        for i in range(time_stamp, len(x_test_scaled)):
            x_test_lstm.append(x_test_scaled[i - time_stamp:i, 0])
            y_test_lstm.append(x_test_scaled[i, 0])

        x_test_lstm, y_test_lstm = np.array(x_test_lstm), np.array(y_test_lstm)

        # Reshape data for LSTM
        x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))
        x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))

        # Create and train the LSTM model
        model = Sequential()
        model.add(LSTM(units=units_1, return_sequences=True, input_shape=(x_train_lstm.shape[1], 1)))
        model.add(LSTM(units=units_2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train_lstm, y_train_lstm, epochs=epochs, batch_size=16, verbose=1)

        # Predictions
        closing_price = model.predict(x_test_lstm)

        # Inverse normalization
        closing_price = scaler.inverse_transform(closing_price)
        y_test = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

        # Calculate RMS
        rms = np.sqrt(np.mean((y_test - closing_price) ** 2))
        print("RMS:", rms)

        # Plot the results
        plt.figure(figsize=(16, 8))
        plt.plot(df.index[-len(y_test):], y_test, color='blue', label='Actual')
        plt.plot(df.index[-len(closing_price):], closing_price, color='green', label='Predictions')
        plt.legend()
        plt.title('LSTM Predicted Result\nEpochs: {}\nComplexity: {}\nRMSE: {:.2f}'.format(epochs, complexity, rms))
        plt.ylabel('Price')
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# Function to predict next time point
def predict_next_time_point():
    try:
        # Preprocess data
        filename = file_entry.get()
        epochs = int(epochs_entry.get())
        complexity = complexity_var.get()
        time_stamp = 20

        # Read file
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        x = df['close'].values.reshape(-1, 1)

        # Normalize values
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(x)

        # Set model complexity
        units_1 = 100
        units_2 = 50

        if complexity == "Simple":
            units_1 = 50
            units_2 = 25
        elif complexity == "Moderate":
            units_1 = 100
            units_2 = 50
        elif complexity == "Complex":
            units_1 = 200
            units_2 = 100

        # Train set
        x_train_lstm, y_train_lstm = [], []
        for i in range(time_stamp, len(x_train_scaled)):
            x_train_lstm.append(x_train_scaled[i - time_stamp:i, 0])
            y_train_lstm.append(x_train_scaled[i, 0])

        x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)

        # Reshape data for LSTM
        x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))

        # Create and train the LSTM model

        batch_size = 16
        model = Sequential()
        model.add(LSTM(units=units_1, return_sequences=True, input_shape=(x_train_lstm.shape[1], 1)))
        model.add(LSTM(units=units_2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=1)

        # Predict future one time point
        # Extract last 20 data points for prediction
        data_to_predict = df['close'][-20:].values.reshape(-1, 1)

        # Normalize data
        scaled_data_to_predict = scaler.transform(data_to_predict)

        # Build feature matrix
        x_to_predict = np.array(
            [scaled_data_to_predict[i:i + time_stamp, 0] for i in range(len(scaled_data_to_predict) - time_stamp + 1)])

        # Reshape data for LSTM
        x_to_predict = np.reshape(x_to_predict, (x_to_predict.shape[0], x_to_predict.shape[1], 1))

        # Use model for prediction
        predicted_price = model.predict(x_to_predict)

        # Inverse normalize predicted price
        predicted_price = scaler.inverse_transform(predicted_price)

        # Show predicted price in messagebox
        messagebox.showinfo("Predicted Price", f"Predicted price for the next time point: {predicted_price[0][0]}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def auto_predict():
    time_stamp = 20
    rms_index = []
    try:
        filename = file_entry.get()
        for i in (25, 50, 75):
            for j in ("Simple", "Moderate", "Complex"):
                epochs = i
                complexity = j

                # Read file
                df = pd.read_csv(filename)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)

                # Feature and target
                x = df['close'].values.reshape(-1, 1)
                y = df['close'].values

                # Split dataset
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

                # Set model complexity
                units_1 = 100
                units_2 = 50
                if complexity == "Simple":
                    units_1 = 50
                    units_2 = 25
                elif complexity == "Moderate":
                    units_1 = 100
                    units_2 = 50
                elif complexity == "Complex":
                    units_1 = 200
                    units_2 = 100

                # Normalize values
                scaler = MinMaxScaler(feature_range=(0, 1))
                x_train_scaled = scaler.fit_transform(x_train)
                x_test_scaled = scaler.transform(x_test)

                # Train set
                x_train_lstm, y_train_lstm = [], []
                for i1 in range(time_stamp, len(x_train_scaled)):
                    x_train_lstm.append(x_train_scaled[i - time_stamp:i, 0])
                    y_train_lstm.append(x_train_scaled[i, 0])

                x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)

                # Test set
                x_test_lstm, y_test_lstm = [], []
                for i2 in range(time_stamp, len(x_test_scaled)):
                    x_test_lstm.append(x_test_scaled[i - time_stamp:i, 0])
                    y_test_lstm.append(x_test_scaled[i, 0])

                x_test_lstm, y_test_lstm = np.array(x_test_lstm), np.array(y_test_lstm)

                # Reshape data for LSTM
                x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))
                x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))

                # Create and train the LSTM model
                model = Sequential()
                model.add(LSTM(units=units_1, return_sequences=True, input_shape=(x_train_lstm.shape[1], 1)))
                model.add(LSTM(units=units_2))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(x_train_lstm, y_train_lstm, epochs=epochs, batch_size=16, verbose=1)

                # Predictions
                closing_price = model.predict(x_test_lstm)

                # Inverse normalization
                closing_price = scaler.inverse_transform(closing_price)
                y_test = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

                # Calculate RMS
                rms = np.sqrt(np.mean((y_test - closing_price) ** 2))
                rms_index.append(rms)

    except Exception as e:
        print(f"An error occurred: {e}")

    best_index = rms_index.index(min(rms_index))
    print(rms_index)
    print(best_index)
    epochs = 50
    complexity = "Moderate"
    if best_index == 0:
        epochs = 25
        complexity = "Simple"
    elif best_index == 1:
        epochs = 25
        complexity = "Moderate"
    elif best_index == 2:
        epochs = 25
        complexity = "Complex"
    elif best_index == 3:
        epochs = 50
        complexity = "Simple"
    elif best_index == 4:
        epochs = 50
        complexity = "Moderate"
    elif best_index == 5:
        epochs = 50
        complexity = "Complex"
    elif best_index == 6:
        epochs = 75
        complexity = "Simple"
    elif best_index == 7:
        epochs = 75
        complexity = "Moderate"
    elif best_index == 8:
        epochs = 75
        complexity = "Complex"

    print(epochs)
    print(complexity)

    # Read file
    filename = file_entry.get()
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    x = df['close'].values.reshape(-1, 1)

    # Normalize values
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x)

    # Set model complexity
    units_1 = 100
    units_2 = 50

    if complexity == "Simple":
        units_1 = 50
        units_2 = 25
    elif complexity == "Moderate":
        units_1 = 100
        units_2 = 50
    elif complexity == "Complex":
        units_1 = 200
        units_2 = 100

    # Train set
    x_train_lstm, y_train_lstm = [], []
    for i in range(time_stamp, len(x_train_scaled)):
        x_train_lstm.append(x_train_scaled[i - time_stamp:i, 0])
        y_train_lstm.append(x_train_scaled[i, 0])

    x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)

    # Reshape data for LSTM
    x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))

    # Create and train the LSTM model

    batch_size = 16
    model = Sequential()
    model.add(LSTM(units=units_1, return_sequences=True, input_shape=(x_train_lstm.shape[1], 1)))
    model.add(LSTM(units=units_2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict future one time point
    # Extract last 20 data points for prediction
    data_to_predict = df['close'][-20:].values.reshape(-1, 1)

    # Normalize data
    scaled_data_to_predict = scaler.transform(data_to_predict)

    # Build feature matrix
    x_to_predict = np.array(
        [scaled_data_to_predict[i:i + time_stamp, 0] for i in range(len(scaled_data_to_predict) - time_stamp + 1)])

    # Reshape data for LSTM
    x_to_predict = np.reshape(x_to_predict, (x_to_predict.shape[0], x_to_predict.shape[1], 1))

    # Use model for prediction
    predicted_price = model.predict(x_to_predict)

    # Inverse normalize predicted price
    predicted_price = scaler.inverse_transform(predicted_price)

    # Show predicted price in messagebox
    messagebox.showinfo("Predicted Price", f"Predicted price for the next time point: {predicted_price[0][0]}")
# Function to quit the application


def quit_application():
    root.destroy()


# Create GUI window
root = tk.Tk()
root.title("Stock price predictor")

# Add widgets to the window
file_label = tk.Label(root, text="Select Dataset File:")
file_label.grid(row=0, column=0)
file_entry = tk.Entry(root)
file_entry.grid(row=0, column=1)
browse_button = tk.Button(root, text="Browse", command=lambda: file_entry.insert(tk.END, filedialog.askopenfilename()))
browse_button.grid(row=0, column=2)

epochs_label = tk.Label(root, text="Enter Epochs:")
epochs_label.grid(row=1, column=0)
epochs_entry = tk.Entry(root)
epochs_entry.grid(row=1, column=1)

complexity_label = tk.Label(root, text="Select Units Complexity Level:")
complexity_label.grid(row=2, column=0)
complexity_var = tk.StringVar(root)
complexity_var.set("Simple")
complexity_option = tk.OptionMenu(root, complexity_var, "Simple", "Moderate", "Complex")
complexity_option.grid(row=2, column=1)

manual_label = tk.Label(root, text="Manual Mode")
manual_label.grid(row=3, column=0)
train_button = tk.Button(root, text="Train Model", command=train_model, fg='green')
train_button.grid(row=3, column=1)
predict_button = tk.Button(root, text="Predict", command=predict_next_time_point, fg='green')
predict_button.grid(row=3, column=2)

auto_label = tk.Label(root, text="Auto Mode")
auto_label.grid(row=4, column=0)
auto_predict_button = tk.Button(root, text="Automatic Predict", command=auto_predict)
auto_predict_button.grid(row=4, column=1)

quit_button = tk.Button(root, text="Quit", command=quit_application, fg='red')
quit_button.grid(row=5, column=1)

# Run the GUI loop
root.mainloop()
