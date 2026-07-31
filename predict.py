import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")

print("SmartFactory AI Prediction System")
print("Model loaded successfully!")

# Example machine sensor readings
machine = pd.DataFrame(
    [[300.5, 309.8, 1345, 62.7, 153]],
    columns=[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
)

print("\nMachine Sensor Data:")
print(machine)

# Make prediction
prediction = model.predict(machine)

print("\nPrediction:")

if prediction[0] == 1:
    print("WARNING: Machine failure predicted!")
else:
    print("Machine operating normally.")