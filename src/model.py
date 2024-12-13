import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv('data/preprocessed_network_traffic_data.csv')  
df['sttime'] = pd.to_datetime(df['sttime'], format='%Y-%m-%d %H:%M:%S')
df.set_index('sttime', inplace=True)
df['sbytes_lag1'] = df['sbytes'].shift(1)
df['sbytes_lag2'] = df['sbytes'].shift(2)
df['sbytes_lag3'] = df['sbytes'].shift(3)
df.dropna(inplace=True)
train_size = int(len(df) * 0.59)
train, test = df.iloc[:train_size], df.iloc[train_size:]
X_train = train[['sbytes_lag1', 'sbytes_lag2', 'sbytes_lag3']]
y_train = train['sbytes']
X_test = test[['sbytes_lag1', 'sbytes_lag2', 'sbytes_lag3']]
y_test = test['sbytes']
try:
    model = joblib.load('model/random_forest_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("No pre-trained model found. Training a new one...")
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model/random_forest_model.pkl')
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
ax1.plot(test.index, y_test, label='Actual Traffic (Last 24 Hours)', linestyle='-', color='blue', marker='o', alpha=0.7)
ax1.plot(test.index, predictions, label='Predicted Traffic (Tomorrow)', linestyle='--', color='red', marker='x', alpha=0.7)
ax1.set_title('Network Traffic Prediction for Tomorrow (Full View)', fontsize=14)
ax1.set_ylabel('Traffic Volume (Bytes)', fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
threshold = y_test.quantile(0.70)  
rush_mask = y_test > threshold
ax2.plot(test.index[rush_mask], y_test[rush_mask], linestyle='-', color='blue', marker='o', alpha=0.7, label='Actual Traffic (Rush Hours)')
ax2.plot(test.index[rush_mask], predictions[rush_mask], linestyle='--', color='red', marker='x', alpha=0.7, label='Predicted Traffic (Rush Hours)')
ax2.set_title('Magnified View of Rush Hour Traffic (Top 15% of Traffic)', fontsize=14)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Traffic Volume (Bytes)', fontsize=12)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
output = pd.DataFrame({'sttime': test.index, "Actual Traffic": y_test, 'Predicted Traffic': predictions})
output.to_csv('result/next_day_predictions.csv', index=False)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
