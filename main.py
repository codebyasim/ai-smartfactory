import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load manufacturing machine data
data = pd.read_csv("data/ai4i2020.csv")

# Display the first five machines
print(data.head())

# Show the size of the dataset
print("\nDataset size:")
print(data.shape)

# Show all column names
print("\nColumns:")
print(data.columns)

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Select 5 important sensor columns
sensor_data = data[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
]

print("\nMachine failure counts:")
print(data["Machine failure"].value_counts())

# Display basic statistics
print("\nSensor statistics:")
print(sensor_data.describe())

# Visualise machine failures
data["Machine failure"].value_counts().plot(kind="bar")

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")

plt.show()

print("SmartFactory AI")
print("Libraries loaded successfully!")