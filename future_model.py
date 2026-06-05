import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# ===== Load Data =====
df = pd.read_csv("USD-INR.csv", parse_dates=["Date"])
df = df.sort_values("Date")
df = df[["Date", "Close"]].rename(columns={"Close": "rate"})
df = df.set_index("Date")

# ===== Feature Engineering =====
df["rate_shift_1"] = df["rate"].shift(1)
df["rate_shift_7"] = df["rate"].shift(7)
df["rate_shift_30"] = df["rate"].shift(30)

df = df.dropna()

# ===== Scaling =====
scaler = MinMaxScaler()
X = scaler.fit_transform(df[["rate_shift_1","rate_shift_7","rate_shift_30"]])
y = df["rate"].values

# ===== Train model =====
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# ===== Save model and scaler =====
joblib.dump(model, "model_future.pkl")
joblib.dump(scaler, "scaler_future.gz")

print("Model training complete! Files saved: model_rf.pkl & scaler_rf.gz")
