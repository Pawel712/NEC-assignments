import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MyNeuralNetwork:
    def __init__(self, num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage):
        self.L = num_layers
        self.n = num_units
        self.num_epochs = num_epochs
        self.eta = learning_rate
        self.alpha = momentum
        self.fact = activation_function
        self.validation_percentage = validation_percentage
        self.xi = [np.zeros(layer_units) for layer_units in num_units]
        self.h = [np.zeros(layer_units) for layer_units in num_units]
        self.w = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
        self.theta = [np.zeros(layer_units) for layer_units in num_units]
        self.delta = [np.zeros(layer_units) for layer_units in num_units]
        self.d_w = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
        self.d_theta = [np.zeros(layer_units) for layer_units in num_units]
        self.d_w_prev = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
        self.d_theta_prev = [np.zeros(layer_units) for layer_units in num_units]
        self.training_error = []
        self.validation_error = []

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones_like(x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - x**2

    def activation(self, x):
        if self.fact == 'sigmoid':
            return self.sigmoid(x)
        elif self.fact == 'relu':
            return self.relu(x)
        elif self.fact == 'linear':
            return self.linear(x)
        elif self.fact == 'tanh':
            return self.tanh(x)

    def activation_derivative(self, x):
        if self.fact == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.fact == 'relu':
            return self.relu_derivative(x)
        elif self.fact == 'linear':
            return self.linear_derivative(x)
        elif self.fact == 'tanh':
            return self.tanh_derivative(x)

    def feed_forward(self, sample):
        self.xi[0] = sample
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.xi[l] = self.activation(self.h[l])

    def backpropagate(self, target):
        self.delta[self.L - 1] = self.activation_derivative(self.xi[self.L - 1]) * (self.xi[self.L - 1] - target)
        for l in range(self.L - 2, 0, -1):
            self.delta[l] = self.activation_derivative(self.xi[l]) * np.dot(self.w[l + 1].T, self.delta[l + 1])

    def update_weights(self):
        for l in range(1, self.L):
            self.d_w[l] = -self.eta * np.outer(self.delta[l], self.xi[l - 1]) + self.alpha * self.d_w_prev[l]
            self.d_theta[l] = self.eta * self.delta[l] + self.alpha * self.d_theta_prev[l]
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l] = self.d_theta[l]

    def calculate_total_error(self, X, y):
        total_error = 0.0
        for i in range(X.shape[0]):
            self.feed_forward(X[i])
            total_error += 0.5 * np.sum((self.xi[self.L - 1] - y[i]) ** 2)
        return total_error

    def fit(self, X, y):
        # Split data into training and validation sets
        n_samples = X.shape[0]
        if self.validation_percentage > 0:
            n_train = int(n_samples * (1.0 - self.validation_percentage))
            X_train = X[:n_train]
            y_train = y[:n_train]
            X_val = X[n_train:]
            y_val = y[n_train:]
        else:
            X_train = X
            y_train = y
            X_val = np.array([])
            y_val = np.array([])

        for epoch in range(self.num_epochs):
            for i in range(X_train.shape[0]):
                sample = X_train[i]
                target = y_train[i]

                self.feed_forward(sample)
                self.backpropagate(target)
                self.update_weights()

            # Calculate training error
            train_error = self.calculate_total_error(X_train, y_train)
            self.training_error.append(train_error)

            # Calculate validation error
            if X_val.shape[0] > 0:
                val_error = self.calculate_total_error(X_val, y_val)
                self.validation_error.append(val_error)

    def predict(self, X):
        predictions = []
        for sample in X:
            self.feed_forward(sample)
            predictions.append(self.xi[self.L - 1].copy())
        return np.array(predictions)

    # Function to retrieve epoch-wise loss data in the required format
    def get_epoch_losses(self):
        epochs = list(range(1, self.num_epochs + 1))
        return np.column_stack((epochs, self.training_error, self.validation_error))

    # Modify the loss_epochs() function to use get_epoch_losses()
    def loss_epochs(self):
        return self.get_epoch_losses()

# Function to train a neural network on a given dataset
def train_neural_network(dataset, input_columns, output_column, num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage):
    X = dataset[input_columns].values
    y = dataset[output_column].values.reshape(-1, 1)

    nn = MyNeuralNetwork(num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage)
    
    nn.fit(X, y)
    
    X_test = X  # You can use a different test dataset if needed
    predictions = nn.predict(X_test)

    return nn, predictions

# Load the three datasets
#dataset1 = pd.read_csv('modified_A1-turbine.csv')
#dataset2 = pd.read_csv('A1-synthetic.csv')
dataset3 = pd.read_csv('A1-real_estate.csv')

# Define input and output columns for each dataset
#input_columns1 = ['height_over_sea_level', 'fall', 'net', 'fall_1', 'flow']
#output_column1 = 'power_of_hydroelectrical_turbine'

#input_columns2 = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
#output_column2 = 'z'

input_columns3 = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
output_column3 = 'Y house price of unit area'

# Define neural network configurations for each dataset

#num_units1 = [len(input_columns1)] + [10, 5, 1]  # Adjust the architecture as needed
#num_units2 = [len(input_columns2)] + [10, 5, 1]
num_units3 = [len(input_columns3)] + [10, 5, 1]

num_layers = 4
num_epochs = 1000
learning_rate = 0.1
momentum = 0.0
activation_function = "sigmoid"
validation_percentage = 0.2

# Train the neural network for each dataset
#nn1, predictions1 = train_neural_network(dataset1, input_columns1, output_column1, num_layers, num_units1, num_epochs, learning_rate, momentum, activation_function, validation_percentage)

#nn2, predictions2 = train_neural_network(dataset2, input_columns2, output_column2, num_layers, num_units2, num_epochs, learning_rate, momentum, activation_function, validation_percentage)

nn3, predictions3 = train_neural_network(dataset3, input_columns3, output_column3, num_layers, num_units3, num_epochs, learning_rate, momentum, activation_function, validation_percentage)

# For dataset 1
#loss_data1 = nn1.loss_epochs()

# For dataset 2 
#ta2 = nn2.loss_epochs()

# For dataset 3
loss_data3 = nn3.loss_epochs()

# Define headers for the loss data CSV file
loss_headers = ['Epochs', 'Training Loss', 'Validation Loss']

# Combine headers with the loss dataset 1
#loss_data_with_headers = np.vstack([loss_headers, loss_data1])

# Combine headers with the loss dataset 2
#loss_data_with_headers = np.vstack([loss_headers, loss_data2])

# Combine headers with the loss dataset 3
loss_data_with_headers = np.vstack([loss_headers, loss_data3])

# Save the loss data along with headers to a CSV file
np.savetxt("loss_data_real_estate.csv", loss_data_with_headers, delimiter=",", fmt='%s')

# Save predictions to files
#np.savetxt("predictions_dataset1.csv", predictions1, delimiter=",")
#np.savetxt("predictions_dataset2.csv", predictions2, delimiter=",")
np.savetxt("predictions_real_estate.csv", predictions3, delimiter=",")

# Generate line plots for training and validation losses vs. epochs
plt.figure(figsize=(8, 6))
plt.plot(loss_data3[:, 0], loss_data3[:, 1], label='Training Loss')
plt.plot(loss_data3[:, 0], loss_data3[:, 2], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs (Real Estate)')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot_real_estate.png')
plt.show()

# Assuming predictions2 contains predicted values and dataset2 contains the real values
real_values = dataset3[output_column3].values.reshape(-1, 1)

# Generate scatter plot for real values vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(real_values, predictions3, alpha=0.5)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Correlation between Real and Predicted Values (Real Estate)')
plt.grid(True)

plt.savefig('correlation_plot_real_estate.png')
plt.show() 



