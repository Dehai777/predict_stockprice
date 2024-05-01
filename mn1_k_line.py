import pandas as pd
import mplfinance as mpf

mn1 = pd.read_csv('archive/MN1/SBUX.US_MN1.csv')
# draw k_line
mn1.index = pd.to_datetime(mn1['datetime'])
mn1.sort_index(inplace=True)
mpf.plot(mn1, type='candle', style='charles', volume=True)
