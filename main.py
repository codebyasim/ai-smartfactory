import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

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

# Compare sensor averages for failed and non-failed machines
failure_comparison = data.groupby("Machine failure")[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
].mean()

print("\nAverage sensor values by machine failure:")
print(failure_comparison)

# Correlation with machine failure
correlation_data = data[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]
]

# Calculate correlations btween -1 and +1
correlations = correlation_data.corr()

print("\nCorrelation with machine failure:")
print(correlations["Machine failure"])

# Features used to predict machine failure
X = data[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
]

# Target
y = data["Machine failure"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

# Create  Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

# Make predictions using test data
y_pred = model.predict(X_test)

print("\nFirst 20 predictions:")
print(y_pred[:20])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Show feature importance
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

print("SmartFactory AI")
print("Libraries loaded successfully!")