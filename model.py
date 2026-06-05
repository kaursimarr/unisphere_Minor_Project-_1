# train_model.py (final 4-feature model, no Research)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import json

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("Admission_Predict.csv")
df.columns = df.columns.str.strip()

required_cols = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "CGPA",
    "Chance of Admit"
]

# Validate required columns
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column in dataset: {col}")

# Keep only required data
df = df[required_cols].dropna()

# Target fix (if in 0-100 scale)
y = df["Chance of Admit"]
if y.max() > 1.01:
    y = y / 100.0

# Features (4)
X = df[["GRE Score", "TOEFL Score", "University Rating", "CGPA"]]

# ---------------------------
# Scale only GRE, TOEFL, CGPA
# ---------------------------
cols_to_scale = ["GRE Score", "TOEFL Score", "CGPA"]
scaler = StandardScaler()
X_scaled_part = scaler.fit_transform(X[cols_to_scale])

# Reconstruct full matrix: [scaled GRE, scaled TOEFL, rating, scaled CGPA]
import numpy as np
X_scaled = np.column_stack([
    X_scaled_part[:,0],          # GRE
    X_scaled_part[:,1],          # TOEFL
    X["University Rating"].values, # NOT scaled
    X_scaled_part[:,2]           # CGPA
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)

print("Model trained successfully!")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# ---------------------------
# Save model + scaler + metadata
# ---------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_features.json", "w") as f:
    json.dump({
        "order": ["GRE", "TOEFL", "Rating", "CGPA"],
        "scaled": ["GRE", "TOEFL", "CGPA"]
    }, f, indent=4)

print("✅ Saved: model.pkl, scaler.pkl, model_features.json")
