import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Read CSV file
df = pd.read_csv("dataset/sales_data.csv")

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

# Create new features
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month

# Input and Output
X = df[["Festival", "Inflation", "Temperature", "Day", "Month"]]
y = df["Sales"]

# Train Model
model = RandomForestRegressor()
model.fit(X, y)

# Save Model
joblib.dump(model, "model/sales_model.pkl")

print("Model Trained Successfully")