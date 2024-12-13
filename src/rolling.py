import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/preprocessed_network_traffic_data.csv')
df['sttime'] = pd.to_datetime(df['sttime'], format='%Y-%m-%d %H:%M:%S')
df.set_index('sttime', inplace=True)
target_variable = 'sbytes'
rolling_window = 30
rolling_mean = df[target_variable].rolling(window=rolling_window).mean()
rolling_std = df[target_variable].rolling(window=rolling_window).std()
plt.figure(figsize=(14, 7))
plt.plot(df.index, df[target_variable], label='Original Data', color='blue')
plt.plot(df.index, rolling_mean, label='Rolling Mean', color='red')
plt.plot(df.index, rolling_std, label='Rolling Std (Variance)', color='black')
plt.title(f'Rolling Mean & Variance (Window={rolling_window})')
plt.xlabel('date and time')
plt.ylabel('bytes sent (kb/s)')
plt.legend()
plt.show()
