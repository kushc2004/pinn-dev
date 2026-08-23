import pandas as pd
cir2=pd.read_csv(r'C:\Users\HP\Desktop\Siemens\Pinns on climatix\427cir2.csv')

cir2['date'] = pd.to_datetime(cir2['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
cir2.set_index('date', inplace=True)
cir2.columns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
X = cir2[['Circuit 2_ EEV1 Open Valve', 'Circuit 2_ Compressor Value', 'Circuit 2_ Condenser Temperature',
          'Circuit 2_ Condeser Fan Value Coil 1', 'Circuit 2_ Condenser Pressure Probe']]
y = cir2['Circuit 2_ Compressor 1 Active Power']


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# Assuming cir2 is your DataFrame with necessary data
X = cir2[['Circuit 2_ EEV1 Open Valve', 'Circuit 2_ Compressor Value', 'Circuit 2_ Condenser Temperature',
          'Circuit 2_ Condeser Fan Value Coil 1', 'Circuit 2_ Condenser Pressure Probe']]
y = cir2['Circuit 2_ Compressor 1 Active Power']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Building the model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Fitting the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Predicting on the test set
y_pred = model.predict(X_test)

# Rescaling the predictions and test data back to original scale
y_pred_rescaled = scaler_y.inverse_transform(y_pred).flatten()
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Metrics calculations on rescaled values
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"R^2 Score: {r2}")
print(f"MAE (Rescaled): {mae}")
print(f"MAPE (Rescaled): {mape}")
print(f"RMSE (Rescaled): {rmse}")

# Creating the DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual Power': y_test_rescaled,
    'Predicted Power': y_pred_rescaled,
    'Difference': y_test_rescaled - y_pred_rescaled
})

# Saving results to a CSV
results_df.to_csv('predicted_vs_actual_power.csv', index=False)

# Outlier detection with a threshold of 10
threshold = 2.5 * np.std(results_df['Difference'])
print(threshold)
outliers_df = results_df[np.abs(results_df['Difference']) > threshold]

# Outlier statistics
num_outliers = len(outliers_df)
outliers_max = outliers_df['Difference'].max()
outliers_min = outliers_df['Difference'].min()
outliers_avg = outliers_df['Difference'].mean()
outliers_spread = outliers_max - outliers_min

print(f"Number of outliers: {num_outliers}")
print(f"Maximum Outlier: {outliers_max}")
print(f"Minimum Outlier: {outliers_min}")
print(f"Average Outlier: {outliers_avg}")
print(f"Spread of Outliers: {outliers_spread}")

# Display a sample of the outliers
print(outliers_df.head())
