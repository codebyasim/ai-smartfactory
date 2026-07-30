import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load manufacturing machine data
data = pd.read_csv("data/ai4i2020.csv")

# Display the first five machines
print(data.head())

print("SmartFactory AI")
print("Libraries loaded successfully!")