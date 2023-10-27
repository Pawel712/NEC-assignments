import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the datasets
dataset1 = pd.read_csv('A1-turbine.csv')
dataset2 = pd.read_csv('A1-synthetic.csv')
dataset3 = pd.read_csv('A1-real_estate.csv')

# Create a Min-Max scaler for input and output variables
scaler = MinMaxScaler()

# Define input and output columns for each dataset
# Dataset 1 column names
input_columns1 = ['height_over_sea_level', 'fall', 'net', 'fall_1', 'flow']
output_column1 = 'power_of_hydroelectrical_turbine'

# Dataset 2 column names
input_columns2 = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
output_column2 = 'z'

# Dataset 3 column names
input_columns3 = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
output_column3 = 'Y house price of unit area'

# Impute NaN values with the mean for each dataset
# Got a runtime warning for missing values in Datasets. So I used Impute NaN Values to continue with min max scaling.: To handle NaN values in the datasets and impute them with the mean, you can use the fillna method.

for column in input_columns1:
    dataset1[column] = dataset1[column].fillna(dataset1[column].mean())

for column in input_columns2:
    dataset2[column] = dataset2[column].fillna(dataset2[column].mean())

for column in input_columns3:
    dataset3[column] = dataset3[column].fillna(dataset3[column].mean())

# Normalize input variables in Dataset 1
dataset1[input_columns1] = scaler.fit_transform(dataset1[input_columns1])

# Normalize input variables in Dataset 2
dataset2[input_columns2] = scaler.fit_transform(dataset2[input_columns2])

# Normalize input variables in Dataset 3
dataset3[input_columns3] = scaler.fit_transform(dataset3[input_columns3])

# Normalize output variables
# For Dataset 1
output_scaler1 = MinMaxScaler()
dataset1[output_column1] = output_scaler1.fit_transform(dataset1[[output_column1]])

# For Dataset 2
output_scaler2 = MinMaxScaler()
dataset2[output_column2] = output_scaler2.fit_transform(dataset2[[output_column2]])

# For Dataset 3
output_scaler3 = MinMaxScaler()
dataset3[output_column3] = output_scaler3.fit_transform(dataset3[[output_column3]])

# Now you have all three datasets normalized using Min-Max scaling with NaN values imputed
# The output variables are also normalized and saved in separate scalers

# To access the normalized data, you can use dataset1, dataset2, and dataset3

# For example, to access the first 5 rows of each dataset:
print("First 5 rows of Dataset 1:")
print(dataset1.head())

print("First 5 rows of Dataset 2:")
print(dataset2.head())

print("First 5 rows of Dataset 3:")
print(dataset3.head())

# Save the normalized datasets to separate CSV files
dataset1.to_csv('normalized_dataset1.csv', index=False)
dataset2.to_csv('normalized_dataset2.csv', index=False)
dataset3.to_csv('normalized_dataset3.csv', index=False)
