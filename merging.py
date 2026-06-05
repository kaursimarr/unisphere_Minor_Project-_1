import pandas as pd

# -----------------------------
# 1. Load datasets
# -----------------------------
indian_students = pd.read_csv("IndianStudentsAbroad.csv")
cost_living = pd.read_csv("Cost_of_Living_Index_by_Country_2024.csv")
tuition = pd.read_csv("International_Education_Costs.csv")

# QS dataset must use latin1 encoding
reputation = pd.read_csv(
    "QS World University Rankings 2025 (Top global universities).csv",
    encoding="latin1",
    engine="python"
)

# -----------------------------
# 2. Clean column names
# -----------------------------
for df_ in [indian_students, cost_living, tuition, reputation]:
    df_.columns = df_.columns.str.strip().str.lower().str.replace(" ", "_")

# Your QS dataset uses: institution_name
if "institution_name" in reputation.columns:
    reputation = reputation.rename(columns={"institution_name": "university"})

# -----------------------------
# 3. Perform merges
# -----------------------------
df = (
    tuition.merge(cost_living, on="country", how="left")
           .merge(indian_students, on="country", how="left")
           .merge(reputation, on="university", how="left")
)

# -----------------------------
# 4. Fill missing values
# -----------------------------
for col in ["visa_fee_usd", "insurance_usd"]:
    if col not in df.columns:
        df[col] = 0
    else:
        df[col] = df[col].fillna(0)

df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# 5. Feature Engineering
# -----------------------------

# Total cost computation
df["total_cost"] = (
    df["tuition_usd"] * df["duration_years"]
    + df["rent_usd"] * df["duration_years"]
    + df["visa_fee_usd"]
    + df["insurance_usd"]
)

# ROI Score
emp_rep = df.get("employer_reputation_score", df.get("employer_reputation", 1))
emp_out = df.get("employment_outcomes_score", df.get("employment_outcomes", 1))

df["roi_score"] = (emp_rep * emp_out) / (df["total_cost"] + 1)

# Normalizations
acad_rep = df.get("academic_reputation_score", df.get("academic_reputation", 0))

df["roi_norm"] = df["roi_score"] / df["roi_score"].max()
df["acad_norm"] = acad_rep / acad_rep.max()
df["cost_norm"] = 1 - (df["total_cost"] / df["total_cost"].max())

# -----------------------------
# 6. Export final master dataset
# -----------------------------
df.to_csv("merge_data.csv", index=False)
print("✅ merge_data.csv created successfully!")
