import pandas as pd

# Load your master CSV
df = pd.read_csv("merge_data.csv")

# Ensure rank_2025 is numeric
df["rank_2025"] = pd.to_numeric(df["rank_2025"], errors="coerce")

# Create new column with default value
df["university_rating"] = 1   # fallback

# Apply your rating logic
df.loc[df["rank_2025"] <= 100, "university_rating"] = 5
df.loc[(df["rank_2025"] > 100) & (df["rank_2025"] <= 200), "university_rating"] = 4
df.loc[(df["rank_2025"] > 200) & (df["rank_2025"] <= 300), "university_rating"] = 3
df.loc[(df["rank_2025"] > 300) & (df["rank_2025"] <= 400), "university_rating"] = 2
df.loc[df["rank_2025"] > 400, "university_rating"] = 1

# Save updated file
df.to_csv("master_data_updated.csv", index=False)

print(" university_rating successfully added!")
