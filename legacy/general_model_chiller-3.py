import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Load the model
model = tf.keras.models.load_model('.\\SecuFEx_2024-08-07_16-03-27_9549008142397822462\\sre_water_chiller_regression\\models\\regression_model_keras\\data\\model')
print(model.summary())
print(model.input_shape)
print(model.inputs)
variables = model.trainable_variables
for var in variables:
    print(f"Name: {var.name}, Shape: {var.shape}")

# Load and prepare the dataset
df = pd.read_csv('cleaned_chiller-3.csv')

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32)

# Predict and rescale the predictions
y_pred = model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Predict the values
y_pred = model.predict(X_test)

# Rescale the predictions
y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate performance metrics
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

print(f"R2 Score: {r2}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")

# Calculate the difference between actual and predicted values
differences = y_test_rescaled - y_pred_rescaled
# Calculate the difference between actual and predicted values

# Identify and analyze outliers
threshold = 2.5 * np.std(differences)
print(f'threshold is {threshold}')

outliers = differences[np.abs(differences) > threshold]

# Calculate outlier statistics
num_outliers = len(outliers)
max_outlier = np.max(outliers)
min_outlier = np.min(outliers)
avg_outlier = np.mean(outliers)
spread_outliers = max_outlier - min_outlier

print(f"Number of Outliers: {num_outliers}")
print(f"Maximum Outlier: {max_outlier}")
print(f"Minimum Outlier: {min_outlier}")
print(f"Average Outlier: {avg_outlier}")
print(f"Spread of Outliers: {spread_outliers}")

# Save actual, predicted, and difference values to a CSV file
results_df = pd.DataFrame({
    'Actual Motor KW': y_test_rescaled,
    'Predicted Motor KW': y_pred_rescaled,
    'Difference': differences
})
results_df.to_csv('predicted_vs_actual_motor_kw_Chiller3_general__1.csv', index=False)

# Save outlier data to a CSV file
outliers_df = pd.DataFrame({
    'Outlier Difference': outliers
})
outliers_df.to_csv('outliers_motor_kw_Chiller3_general_model__1.csv', index=False)
