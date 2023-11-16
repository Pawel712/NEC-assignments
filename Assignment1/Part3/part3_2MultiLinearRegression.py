import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

# Function to train a multilinear regression model on a given dataset
def train_multilinear_regression(dataset, input_columns, output_column):
    X = dataset[input_columns].values
    y = dataset[output_column].values.reshape(-1, 1)

    # Your multilinear regression training code here...
    # Assuming you're using scikit-learn LinearRegression for multilinear regression
    from sklearn.linear_model import LinearRegression

    multi_reg_model = LinearRegression()
    multi_reg_model.fit(X, y)

    multi_reg_predictions = multi_reg_model.predict(X)

    return multi_reg_model, multi_reg_predictions


# Load the dataset
#dataset1 = pd.read_csv('A1-synthetic.csv')
#dataset2 = pd.read_csv('modified_A1-turbine.csv')
dataset3 = pd.read_csv('A1-real_estate.csv')

# Define input and output columns for dataset 2
#input_columns1 = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
#output_column1 = 'z'

#input_columns2 = ['height_over_sea_level', 'fall', 'net', 'fall_1', 'flow']
#output_column2 = 'power_of_hydroelectrical_turbine'

input_columns3 = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
output_column3 = 'Y house price of unit area'

# Neural network configurations
num_layers = 4
num_units2 = [len(input_columns3)] + [10, 5, 1]
num_epochs = 1000
learning_rate = 0.1
momentum = 0.0
activation_function = "sigmoid"
validation_percentage = 0.2

# Training the multilinear regression model for dataset 1
#multi_reg_model, multi_reg_predictions = train_multilinear_regression(dataset1, input_columns1, output_column1)

# Training the multilinear regression model for dataset 2
#multi_reg_model, multi_reg_predictions = train_multilinear_regression(dataset2, input_columns2, output_column2)

# Training the multilinear regression model for dataset 3
multi_reg_model, multi_reg_predictions = train_multilinear_regression(dataset3, input_columns3, output_column3)

# Assuming predictions2 contains predicted values and dataset1 contains the real values
#real_values = dataset1[output_column1].values.reshape(-1, 1)

# Assuming predictions2 contains predicted values and dataset1 contains the real values
#real_values = dataset1[output_column1].values.reshape(-1, 1)

# Assuming predictions2 contains predicted values and dataset1 contains the real values
real_values = dataset3[output_column3].values.reshape(-1, 1)

# Calculate MAPE for dataset 2
mape2 = calculate_mape(real_values, multi_reg_predictions)
print(f"MAPE for Dataset turbine: {mape2:.2f}%")

# Visualize the scatter plot for dataset 2
plt.figure(figsize=(8, 6))
plt.scatter(real_values, multi_reg_predictions, alpha=0.5)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Correlation between Real and Predicted Values (Dataset realEstate, multi linear regression)')
plt.grid(True)
plt.plot([real_values.min(), real_values.max()], [real_values.min(), real_values.max()], 'k--', lw=2) # diagonal line
plt.savefig('correlation_plot_realEstate_multi_linear.png')
plt.show()

#Multilinear regression output:
np.savetxt("realEstatepredictionMultiLinearRegression.csv", multi_reg_predictions, delimiter=",")


