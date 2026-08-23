from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



cir1 = pd.read_csv(r'./427cir1.csv')

feature_columns = [
    'Circuit 1_ EEV1 Open Valve', 'Circuit 1_ Compressor Value', 'Circuit 1_ Condenser Temperature',
          'Circuit 1_ Condeser Fan Value Coil 1', 'Circuit 1_ Condenser Pressure Probe'
]




df = pd.read_csv(r'./427cir1.csv')

feature_columns = [
    'Circuit 1_ EEV1 Open Valve', 'Circuit 1_ Compressor Value', 'Circuit 1_ Condenser Temperature',
          'Circuit 1_ Condeser Fan Value Coil 1', 'Circuit 1_ Condenser Pressure Probe'
]
target_column = 'Circuit 1_ Compressor 1 Active Power'
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

model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1))

print(model.summary())
print(model.input_shape)
print(model.inputs)

class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # Cast features to float32 as per the original requirement
        eev1_open_valve = tf.cast(X_train[:, 0], tf.float32)
        compressor_value = tf.cast(X_train[:, 1], tf.float32)
        condenser_temperature = tf.cast(X_train[:, 2], tf.float32)
        fan_value_coil = tf.cast(X_train[:, 3], tf.float32)
        condenser_pressure_probe = tf.cast(X_train[:, 4], tf.float32)

    #     # Adjust the formula for the predicted_motor_load accordingly, if needed
    #     predicted_motor_load = (
    #     (compressor_value + 
    #      (compressor_value * (
    #          compressor_value - 
    #          (2.458 - ((2.458 - (compressor_value / (-1.654 / eev1_open_valve))) / compressor_value))
    #      ) / fan_value_coil) - 
    #      (0.750 - (compressor_value / (-1.654 / eev1_open_valve)))
    #     ) / compressor_value
    # )

        # Calculate MSE and physics-based loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        # physics_loss = tf.reduce_mean(tf.square((predicted_motor_load - y_pred) / (1 + tf.abs(predicted_motor_load))))

        # Total loss
        total_loss =  mse_loss

        return total_loss


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=CustomLoss())

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

y_pred = model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred).flatten()
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()


r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"R2 Score: {r2}")

mae_rescaled = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"MAE (Rescaled): {mae_rescaled}")
mape_rescaled = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
print(f"MAPE (Rescaled): {mape_rescaled}")
rmse_rescaled = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"RMSE (Rescaled): {rmse_rescaled}")

results_df = pd.DataFrame({
    'Actual Power': y_test_rescaled,
    'Predicted Power': y_pred_rescaled,
    'Difference': y_test_rescaled - y_pred_rescaled
})
results_df.to_csv('predicted_vs_actual_power.csv', index=False)

threshold = 1.9333689539639565

outliers_df = results_df[np.abs(results_df['Difference']) > threshold]

num_outliers = len(outliers_df)
print(f"Number of outliers: {num_outliers}")

outliers_max = outliers_df['Difference'].max()
outliers_min = outliers_df['Difference'].min()
outliers_avg = outliers_df['Difference'].mean()
outliers_spread = outliers_max - outliers_min

print(f"Maximum Outlier: {outliers_max}")
print(f"Minimum Outlier: {outliers_min}")
print(f"Average Outlier: {outliers_avg}")
print(f"Spread of Outliers: {outliers_spread}")

print(outliers_df.head())

