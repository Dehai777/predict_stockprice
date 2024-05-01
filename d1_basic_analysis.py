import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read file.
d1 = pd.read_csv('archive/D1/SBUX.US_D1.csv')

print('Shape before deleting duplicate values:', d1.shape)
# Removing duplicate rows if any
d1 = d1.drop_duplicates()
print('Shape after deleting duplicate values:', d1.shape)

# Printing sample data
print(d1.head(5))
# d1.info()

# Make the datetime as index.
d1['datetime'] = pd.to_datetime(d1['datetime'])
d1.set_index('datetime', inplace=True)

# Show close
d1_close = d1['close']
plt.plot(d1_close)
# plt.show()

# Add some parameters to help us better understand the data.
d1['H-L'] = d1['high'] - d1['low']
d1['O-C'] = d1['close'] - d1['open']
d1['Std_dev'] = d1['close'].rolling(5).std()
d1_types = d1.dtypes

# Set display.
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Describe data.
d1_describe = d1.describe().T
print(d1_describe)

# Display the count of different values for each item.
print('{0:12}{1:10}{2:6}'.format('column', 'nunique', 'NaN'))
for column in d1:
    print('{0:12}{1:6d}{2:6}'.format(column, d1[column].nunique(), (d1[column] == -1).sum()))

# Show missing value in a barchart.
d1_missing_count = d1.isnull().sum()
plt.rcParams['figure.figsize'] = (15, 8)
d1_miss_p = d1_missing_count.plot.bar()
d1_miss_p.set_ylim(0, 10)
plt.show()

# Display rows with missing data.
missing_rows = d1[d1.isnull().any(axis=1)]
print(missing_rows)

# Use histograms to display each item.
columns_multi = [x for x in list(d1.columns)]
d1.hist(layout=(3, 3), column=columns_multi)
fig = plt.gcf()
fig.set_size_inches(20, 9)
plt.show()

# Display a density plot.
names = columns_multi
d1.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()

# Display periods with significant fluctuations.
d1_f = d1[d1['H-L'] > 2]
d1_f_peryear = d1_f.resample('YE').size()
print(d1_f_peryear)

# Display a heatmap to observe the correlation between the data.
d1_camp = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(d1.corr(), annot=True, cmap='Blues')
plt.show()

# Show Joint Distribution
sns.pairplot(d1, size=1, diag_kind='kde')
plt.show()


