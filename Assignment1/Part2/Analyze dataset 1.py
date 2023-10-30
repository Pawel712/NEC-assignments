import numpy as np
import pandas as pd

# Load the datasets
try:
    dataset1 = pd.read_csv('A1-turbine.csv')
except FileNotFoundError:
    print("Error: 'A1-turbine.csv' file not found.")
    dataset1 = None
except pd.errors.EmptyDataError:
    print("Error: 'A1-turbine.csv' is empty.")
    dataset1 = None

try:
    dataset2 = pd.read_csv('A1-synthetic.csv')
except FileNotFoundError:
    print("Error: 'A1-synthetic.csv' file not found.")
    dataset2 = None
except pd.errors.EmptyDataError:
    print("Error: 'A1-synthetic.csv' is empty.")
    dataset2 = None

try:
    dataset3 = pd.read_csv('A1-real_estate.csv')
except FileNotFoundError:
    print("Error: 'A1-real_estate.csv' file not found.")
    dataset3 = None
except pd.errors.EmptyDataError:
    print("Error: 'A1-real_estate.csv' is empty.")
    dataset3 = None

# Analyze dataset 1 if it was successfully loaded
if dataset1 is not None:
    print("Dataset 1 Summary:")
    print(dataset1.describe())

# Analyze dataset 2 if it was successfully loaded
if dataset2 is not None:
    print("Dataset 2 Summary:")
    print(dataset2.describe())

# Analyze dataset 3 if it was successfully loaded
if dataset3 is not None:
    print("Dataset 3 Summary:")
    print(dataset3.describe())