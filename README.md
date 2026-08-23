# PINN With Symbolic Regression

## Objective :
The objective of a Physics-Informed Neural Network (PINN) is to integrate physical laws, expressed as partial differential equations (PDEs) or other governing equations, directly into the training process of neural networks. This approach ensures that the predictions made by the neural network adhere to the underlying physics of the problem, leading to more accurate and reliable models, especially in scenarios where data is scarce or noisy.
 
## Description of approach:
Utilize genetic optimization in symbolic regression to discover the underlying physics equation from the dataset. 
Develop a neural network model.
incorporate the discovered equations as a loss function within the neural network architecture. 
Train the model on any dataset.
Predict results that adhere to the system's physics.

## Files Description

cleaned_chiller-3.csv - Contains the data of Chiller 3 on which the model will be trained and tested. Use the relative path to use this dataset.

general_model_chiller-3.py - Contains the code for finding chiller motor power using ipae model after training it and predicting the power giving out metrices, details on outliers.

general_model_chiller_extrapolation-3.py - Contains the code for forecasting chiller power on a range using general ipae model.

pinn_model_chiller-3.py - Contains the code for pinn model made by using custom loss function generated using symbolic regression and ipae model.

pinn_model_chiller-3_extrapolation.py - Contains the code for forecasting chiller power on a range using pinn model.

## Requirements

Download the chiller model - sre_water_chiller_regression\\models\\regression_model_keras and use it relative path in the Files

Libraries:
Tensorflow
Pandas
Numpy
Scikit-learn
Matplotlib

Use python run file_name to run a particular file. 