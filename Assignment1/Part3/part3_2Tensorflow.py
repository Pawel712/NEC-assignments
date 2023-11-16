import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

# Function to train a neural network using TensorFlow on a given dataset
def train_neural_network_tensorflow(dataset, input_columns, output_column, num_layers, num_units, num_epochs, learning_rate):
    X = dataset[input_columns].values
    y = dataset[output_column].values.reshape(-1, 1)

    # TensorFlow neural network training code here.
    import tensorflow as tf

    model = tf.keras.Sequential()
    for i in range(1, len(num_units)):
        if i == 1:
            model.add(tf.keras.layers.Dense(num_units[i], input_dim=len(input_columns), activation='relu'))
        else:
            model.add(tf.keras.layers.Dense(num_units[i], activation='relu'))

    model.add(tf.keras.layers.Dense(1))  # Output layer

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    history = model.fit(X, y, epochs=num_epochs, verbose=0)

    tf_nn_model = model
    tf_nn_predictions = model.predict(X)
    tf_nn_history = history

    return tf_nn_model, tf_nn_predictions, tf_nn_history


# Load the dataset
dataset1 = pd.read_csv('A1-synthetic.csv')
#dataset2 = pd.read_csv('modified_A1-turbine.csv')
#dataset3 = pd.read_csv('A1-real_estate.csv')


# Define input and output columns for dataset 2
input_columns1 = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
output_column1 = 'z'

#input_columns2 = ['height_over_sea_level', 'fall', 'net', 'fall_1', 'flow']
#output_column2 = 'power_of_hydroelectrical_turbine'

#input_columns3 = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
#output_column3 = 'Y house price of unit area'

# Neural network configurations
num_layers = 4
num_units2 = [len(input_columns1)] + [10, 5, 1]
num_epochs = 1000
learning_rate = 0.1
momentum = 0.0
activation_function = "sigmoid"
validation_percentage = 0.2

# Training the neural network using TensorFlow for dataset 1
tf_nn_model, tf_nn_predictions, tf_nn_history = train_neural_network_tensorflow(dataset1, input_columns1, output_column1, num_layers, num_units2, num_epochs, learning_rate)

# Training the neural network using TensorFlow for dataset 2
#tf_nn_model, tf_nn_predictions, tf_nn_history = train_neural_network_tensorflow(dataset2, input_columns2, output_column2, num_layers, num_units2, num_epochs, learning_rate)

# Training the neural network using TensorFlow for dataset 3
#tf_nn_model, tf_nn_predictions, tf_nn_history = train_neural_network_tensorflow(dataset3, input_columns3, output_column3, num_layers, num_units2, num_epochs, learning_rate)

# Assuming predictions2 contains predicted values and dataset2 contains the real values
real_values = dataset1[output_column1].values.reshape(-1, 1)

# Calculate MAPE for dataset 2
mape2 = calculate_mape(real_values, tf_nn_predictions)
print(f"MAPE for Dataset 2: {mape2:.2f}%")

# Visualize the scatter plot for dataset 2
plt.figure(figsize=(8, 6))
plt.scatter(real_values, tf_nn_predictions, alpha=0.5)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Correlation between Real and Predicted Values (Dataset turbine, tensorflow)')
plt.grid(True)
plt.plot([real_values.min(), real_values.max()], [real_values.min(), real_values.max()], 'k--', lw=2) # diagonal line
plt.savefig('correlation_plot_Turbine_tensorflow.png')
plt.show()


#Tensorflow output:
np.savetxt("RealestatepredictionTensorflow.csv", tf_nn_predictions, delimiter=",")
np.savetxt("RealestatelossDataset2Tensorflow.csv", tf_nn_history.history['loss'], delimiter=",")


