import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('.\\SecuFEx_2024-08-07_16-03-27_9549008142397822462\\sre_water_chiller_regression\\models\\regression_model_keras\\data\\model')
print(model.summary())

# Load and prepare the dataset
df = pd.read_csv('cleaned_chiller-3.csv')

# Feature and target columns
feature_columns = [
    'Evaporator Inlet Water Temperature',
    'Evaporator Outlet Water Temperature',
    'Evaporator Flow rate',
    'Condensor Refrigerant Pressure',
    'Condensor Inlet Water Temperature',
    'Condensor Outlet Water Temperature',
    'Condensor Flow Rate'
]
target_column = 'Motor KW'
X = df[feature_columns].values
y = df[target_column].values

# Data scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Train the model (if needed)
history = model.fit(X_scaled, y_scaled, epochs=100, batch_size=32)

# Define a range from the dataset for extrapolation
range_min, range_max = np.percentile(y_scaled, [0, 10])  
extrapolation_indices = np.where((y_scaled < range_min) | (y_scaled > range_max))[0]
X_extrapolation = X_scaled[extrapolation_indices]
y_extrapolation = y_scaled[extrapolation_indices]

# Predict on the extrapolation set
y_extrap_pred = model.predict(X_extrapolation)
y_extrap_pred_rescaled = scaler_y.inverse_transform(y_extrap_pred.reshape(-1, 1)).flatten()
y_extrap_rescaled = scaler_y.inverse_transform(y_extrapolation.reshape(-1, 1)).flatten()

# Calculate performance metrics for extrapolation
r2_extrap = r2_score(y_extrap_rescaled, y_extrap_pred_rescaled)
mae_extrap = mean_absolute_error(y_extrap_rescaled, y_extrap_pred_rescaled)
mape_extrap = mean_absolute_percentage_error(y_extrap_rescaled, y_extrap_pred_rescaled)
rmse_extrap = np.sqrt(mean_squared_error(y_extrap_rescaled, y_extrap_pred_rescaled))

print("\nExtrapolation Performance Metrics:")
print(f"R2 Score: {r2_extrap}")
print(f"MAE: {mae_extrap}")
print(f"MAPE: {mape_extrap}")
print(f"RMSE: {rmse_extrap}")

# Plot actual vs predicted for extrapolation
plt.figure(figsize=(10, 6))
plt.scatter(y_extrap_rescaled, y_extrap_pred_rescaled, color='blue', label='Predictions', alpha=0.5)
plt.plot([min(y_extrap_rescaled), max(y_extrap_rescaled)], [min(y_extrap_rescaled), max(y_extrap_rescaled)], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual Motor KW')
plt.ylabel('Predicted Motor KW')
plt.title('Extrapolation: Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and analyze outliers in extrapolation
differences_extrap = y_extrap_rescaled - y_extrap_pred_rescaled
threshold_extrap = 2.5 * np.std(differences_extrap)
outliers_extrap = differences_extrap[np.abs(differences_extrap) > threshold_extrap]

# Print outlier statistics
num_outliers_extrap = len(outliers_extrap)
max_outlier_extrap = np.max(outliers_extrap)
min_outlier_extrap = np.min(outliers_extrap)
avg_outlier_extrap = np.mean(outliers_extrap)
spread_outliers_extrap = max_outlier_extrap - min_outlier_extrap

print(f"Extrapolation Outlier Analysis:")
print(f"Threshold: {threshold_extrap}")
print(f"Number of Outliers: {num_outliers_extrap}")
print(f"Max Outlier: {max_outlier_extrap}")
print(f"Min Outlier: {min_outlier_extrap}")
print(f"Avg Outlier: {avg_outlier_extrap}")
print(f"Spread of Outliers: {spread_outliers_extrap}")

# Save extrapolation results to CSV
results_extrap_df = pd.DataFrame({
    'Actual Motor KW': y_extrap_rescaled,
    'Predicted Motor KW': y_extrap_pred_rescaled,
    'Difference': differences_extrap
})
results_extrap_df.to_csv('extrapolated_vs_actual_motor_kw_general.csv', index=False)

# Save extrapolation outliers to CSV
outliers_extrap_df = pd.DataFrame({
    'Outlier Difference': outliers_extrap
})
outliers_extrap_df.to_csv('outliers_motor_kw_extrapolation_general.csv', index=False)