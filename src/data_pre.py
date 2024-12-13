import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
df = pd.read_csv('data/dataset_net.csv')
df['sttime'] = pd.to_datetime(df['sttime'], infer_datetime_format=True, dayfirst=True)
df.set_index('sttime', inplace=True)
df = df.loc[~df.index.duplicated(keep='first')]
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

def remove_outliers_zscore(df, columns, threshold=3):
    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < threshold]
    return df

columns_to_check = ['sbytes', 'dbytes', 'spkts', 'dpkts']
df = remove_outliers_zscore(df, columns_to_check)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[['sbytes', 'dbytes', 'spkts', 'dpkts', 'sload', 'dload']])
df_scaled = pd.DataFrame(scaled_features, index=df.index, columns=['sbytes', 'dbytes', 'spkts', 'dpkts', 'sload', 'dload'])

df_scaled['dur'] = df['dur']
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] > 0.05:
        print("The series is non-stationary")
    else:
        print("The series is stationary")

check_stationarity(df_scaled['sbytes'])

# Differencing to make the data stationary (if required)
df_scaled['sbytes_diff'] = df_scaled['sbytes'].diff()

# Creating lag features (for models like LSTM)
df_scaled['sbytes_lag1'] = df_scaled['sbytes'].shift(1)
df_scaled['sbytes_lag2'] = df_scaled['sbytes'].shift(2)

# Dropping any NaN values resulting from differencing or lagging
df_scaled.dropna(inplace=True)

# Save the preprocessed DataFrame to a CSV file
df_scaled.to_csv('data/preprocessed_network_traffic_data.csv')

# Print the head of the final DataFrame to verify
print(df_scaled.head())
