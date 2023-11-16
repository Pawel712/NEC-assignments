# Script for adding the missing values in A1-turbine.csv for column 'power_of_hydroelectrical_turbine'

import pandas as pd

# Read the dataset into a DataFrame
# Assuming the dataset is in a CSV file named 'your_dataset.csv'
df = pd.read_csv('A1-turbine.csv')

# Constants
density_of_water = 1000  # kg/m^3
gravity = 9.81  # m/s^2

# Compute the hydroelectrical power and add it to the DataFrame
df['computed_power'] = density_of_water * gravity * df['flow'] * df['fall']

# Update the 'power_of_hydroelectrical_turbine' column with the computed power
df['power_of_hydroelectrical_turbine'] = df['computed_power']

# Drop the temporary 'computed_power' column if you don't need it anymore
df = df.drop(columns=['computed_power'])

# Save the modified DataFrame back to a CSV file
df.to_csv('modified_dataset.csv', index=False)
