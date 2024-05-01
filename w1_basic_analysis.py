import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Read file.
w1 = pd.read_csv('archive/W1/SBUX.US_W1.csv')

# Make the datetime as index.
w1['datetime'] = pd.to_datetime(w1['datetime'])
w1.set_index('datetime', inplace=True)

# Add some parameters to help us better understand the data.
w1['H-L'] = w1['high'] - w1['low']
w1['O-C'] = w1['close'] - w1['open']
w1['Std_dev'] = w1['close'].rolling(5).std()
w1_types = w1.dtypes

# Set display.
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Display periods with significant fluctuations.
w1_f = w1[w1['H-L'] > 3]
w1_f_peryear = w1_f.resample('YE').size()
print(w1_f_peryear)


# Describe data.
w1_describe = w1.describe().T
print(w1_describe)

# Show missing value in a barchart.
w1_missing_count = w1.isnull().sum()
plt.rcParams['figure.figsize'] = (15, 8)
w1_missing_count.plot.bar()
plt.show()

# Display the count of different values for each item.
print('{0:12}{1:10}{2:6}'.format('column', 'nunique', 'NaN'))
for column in w1:
    print('{0:12}{1:6d}{2:6}'.format(column, w1[column].nunique(), (w1[column] == -1).sum()))

# Use histograms to display each item.
columns_multi = [x for x in list(w1.columns)]
w1.hist(layout=(3, 3), column=columns_multi)
fig = plt.gcf()
fig.set_size_inches(20, 9)
plt.show()

# Display a density plot.
names = columns_multi
w1.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()

# Display rows with missing data.
missing_rows = w1[w1.isnull().any(axis=1)]
print(missing_rows)

# Display a heatmap to observe the correlation between the data.
w1_camp = sns.diverging_palette(220,10, as_cmap=True)
sns.heatmap(w1.corr(), annot=True, cmap='Blues')
plt.show()

# Show Joint Distribution
sns.pairplot(w1, size=1, diag_kind='kde')
plt.show()
