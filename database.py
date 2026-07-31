import sqlite3

# Connect to SQLite database
connection = sqlite3.connect("smartfactory.db")

cursor = connection.cursor()

# Create predictions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    air_temperature REAL,
    process_temperature REAL,
    rotational_speed REAL,
    torque REAL,
    tool_wear REAL,
    prediction INTEGER
)
""")

connection.commit()
connection.close()

print("SmartFactory database created successfully!")