import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

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

print(X_train.shape)

# Custom loss function
class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        evaporator_inlet_temp = tf.cast(X_train[:, 1], tf.float32)
        evaporator_outlet_temp = tf.cast(X_train[:, 2], tf.float32)
        evaporator_flow_rate = tf.cast(X_train[:, 3], tf.float32)
        condenser_refrigerant_pressure = tf.cast(X_train[:, 4], tf.float32)
        condenser_inlet_temp = tf.cast(X_train[:, 5], tf.float32)
        condenser_outlet_temp = tf.cast(X_train[:, 6], tf.float32)
        condenser_flow_rate = tf.cast(X_train[:, -1], tf.float32)

        predicted_motor_load = (
            (condenser_outlet_temp * evaporator_inlet_temp +
             (evaporator_inlet_temp * 0.548 -
              (condenser_inlet_temp -
               (evaporator_flow_rate +
                (condenser_flow_rate * condenser_inlet_temp * -0.125) / condenser_outlet_temp)
              )
             )
            ) *
            ((condenser_outlet_temp + 0.490) * (0.455 / condenser_outlet_temp))
        )
        
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        physics_loss = tf.reduce_mean(tf.square(predicted_motor_load - y_pred))
        
        total_loss = 5*physics_loss + mse_loss
        
        return total_loss

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=CustomLoss())

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
y_pred = model.predict(X_test)

# Rescale predictions
y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate metrics
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"R2 Score: {r2}")

mae_rescaled = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"MAE (Rescaled): {mae_rescaled}")
mape_rescaled = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
print(f"MAPE (Rescaled): {mape_rescaled}")
rmse_rescaled = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"RMSE (Rescaled): {rmse_rescaled}")

# Save actual and predicted values to CSV
results_df = pd.DataFrame({
    'Actual Motor KW': y_test_rescaled,
    'Predicted Motor KW': y_pred_rescaled,
    'Difference': y_test_rescaled - y_pred_rescaled
})
results_df.to_csv('predicted_vs_actual_motor_kw_Chiller3_final_10.csv', index=False)

# Identify and analyze outliers
threshold = 8.806006641814667
# print(f'threshold is {threshold}')
outliers_df = results_df[np.abs(results_df['Difference']) > threshold]

# Number of outliers
num_outliers = len(outliers_df)
print(f"Number of outliers: {num_outliers}")

# Outliers statistics
outliers_max = outliers_df['Difference'].max()
outliers_min = outliers_df['Difference'].min()
outliers_avg = outliers_df['Difference'].mean()
outliers_spread = outliers_max - outliers_min

print(f"Maximum Outlier: {outliers_max}")
print(f"Minimum Outlier: {outliers_min}")
print(f"Average Outlier: {outliers_avg}")
print(f"Spread of Outliers: {outliers_spread}")

# Display the first few outliers
print(outliers_df.head())

# Visualize the outliers
plt.figure(figsize=(10, 6))
plt.scatter(outliers_df.index, outliers_df['Actual Motor KW'], color='red', label='Actual Outliers', alpha=0.6)
plt.scatter(outliers_df.index, outliers_df['Predicted Motor KW'], color='blue', label='Predicted Outliers', alpha=0.6)
plt.title('Outliers in Actual vs Predicted Motor KW')
plt.xlabel('Sample Index')
plt.ylabel('Motor KW')
plt.legend()
plt.grid(True)
plt.savefig('./outliers_actual_vs_predicted_motor_kw_Chiller3_final_10.png', dpi=300)
plt.show()

# Visualize the predicted vs actual values with a line plot (for large datasets)
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label='Actual', color='blue', alpha=0.6)
plt.plot(y_pred_rescaled, label='Predicted', color='orange', alpha=0.6)
plt.title('Actual vs Predicted Motor KW (Line Plot)')
plt.xlabel('Samples')
plt.ylabel('Motor KW')
plt.legend()
plt.grid(True)
plt.savefig('./actual_vs_predicted_motor_kw_lineplot_Chiller3_final_10.png', dpi=300)
plt.show()

# Plot the training and validation loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./training_validation_loss_curve_100_Chiller3_final_10.png')
plt.show()
