import joblib
import pandas as pd
import sqlite3

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

print("SmartFactory AI Prediction System")
print("Model loaded successfully!")

# Get sensor readings from user
air_temp = float(input("Enter air temperature [K]: "))
process_temp = float(input("Enter process temperature [K]: "))
rotational_speed = float(input("Enter rotational speed [rpm]: "))
torque = float(input("Enter torque [Nm]: "))
tool_wear = float(input("Enter tool wear [min]: "))

# Create machine data
machine = pd.DataFrame(
    [[
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]],
    columns=[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
)

# Make prediction
prediction = model.predict(machine)

print("\n--- Prediction Result ---")


# Convert prediction to normal Python integer
prediction_value = int(prediction[0])

# Connect to SmartFactory database
connection = sqlite3.connect("smartfactory.db")
cursor = connection.cursor()

# Save machine readings and prediction
cursor.execute("""
INSERT INTO predictions (
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    prediction
)
VALUES (?, ?, ?, ?, ?, ?)
""", (
    air_temp,
    process_temp,
    rotational_speed,
    torque,
    tool_wear,
    prediction_value
))

connection.commit()
connection.close()

print("Prediction saved to database.")

if prediction[0] == 1:
    print("WARNING: Machine failure predicted!")
else:
    print("Machine operating normally.")