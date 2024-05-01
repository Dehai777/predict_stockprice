import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import os
from math import sqrt
from sklearn.metrics import mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w1 = pd.read_csv('archive/W1/SBUX.US_W1.csv')

w1['datetime'] = pd.to_datetime(w1['datetime'])
w1.set_index('datetime', inplace=True)

w1_close = w1['close']
df = w1_close['1998':'2019']
df_test = w1_close['2019':'2024']

# Create model
# 1. `start_p`: Starting AR order.
# 2. `start_q`: Starting MA order.
# 3. `information_criterion`: Criterion for model selection ('aic' or 'bic').
# 4. `test`: Type of stationarity test ('adf' or other).
# 5. `max_p`: Maximum AR order.
# 6. `max_q`: Maximum MA order.
# 7. `m`: Length of seasonal cycle.
# 8. `d`: Order of differencing.
# 9. `seasonal`: Consider seasonality (True/False).
# 10. `start_P`: Starting seasonal order.
# 11. `D`: Seasonal differencing order.
# 12. `trace`: Print debugging info (True/False).
# 13. `error_action`: Action on error ('ignore' or 'warn').
# 14. `suppress_warnings`: Suppress warnings (True/False).
# 15. `stepwise`: Use stepwise search (True/False).
model = pm.auto_arima(df.values, start_p=1, start_q=1,
                      information_criterion='aic',
                      test='adf',
                      max_p=3, max_q=3,
                      m=12,
                      d=1,
                      seasonal=True,
                      start_P=0,
                      D=1,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

# Display train result
print(model.summary())

# Set the period of forcast
n_periods = len(df_test)

# Predict
arima_result = model.predict(n_periods=n_periods)

# Set the datatime as the index of the result
index_of_fc = pd.date_range(start=df_test.index[0], periods=n_periods, freq='W')

# make series for plotting purpose
arima_result_series = pd.Series(arima_result, index=index_of_fc)

rms = sqrt(mean_squared_error(df_test, arima_result_series))
print(f'RMS is {rms}')

# Display plot
plt.plot(arima_result_series.index, arima_result_series.values, color='red', label='Forecast')
plt.plot(df_test.index, df_test.values, color='blue', label='Actual')
plt.title('ARIMA Predicted Result\nRMSE: {:.2f}'.format(rms))
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
