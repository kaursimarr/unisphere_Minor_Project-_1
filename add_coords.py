# add_coords.py
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

INPUT_CSV = "master_data_updated.csv"
OUTPUT_CSV = "master_data_updated.csv"  # overwrite same file

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip().str.lower()

# Make sure we have city & country
if "city" not in df.columns or "country" not in df.columns:
    raise ValueError("master_data_updated.csv must have 'city' and 'country' columns")

# Create columns if missing
if "latitude" not in df.columns:
    df["latitude"] = None
if "longitude" not in df.columns:
    df["longitude"] = None

geolocator = Nominatim(user_agent="study_abroad_uniglobe")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)  # be gentle with the API

# Build a unique list of places to geocode to keep calls small
locations = (
    df[["university", "city", "country"]]
    .astype(str)
    .drop_duplicates()
    .reset_index(drop=True)
)

coords_map = {}

for _, row in locations.iterrows():
    uni = row.get("university", "")
    city = row.get("city", "")
    country = row.get("country", "")

    # Most precise: university + city + country
    query_full = f"{uni}, {city}, {country}".strip(", ")
    query_city = f"{city}, {country}".strip(", ")

    lat, lon = None, None
    try:
        loc = geocode(query_full)
        if loc is None and city and country:
            loc = geocode(query_city)
        if loc is not None:
            lat, lon = loc.latitude, loc.longitude
    except Exception as e:
        print("Error geocoding", query_full, "->", e)

    key = (uni, city, country)
    coords_map[key] = (lat, lon)
    print(f"{query_full} -> {lat}, {lon}")
    # small delay already handled by RateLimiter

# Apply back to df
for idx, row in df.iterrows():
    key = (
        str(row.get("university", "")),
        str(row.get("city", "")),
        str(row.get("country", "")),
    )
    if key in coords_map:
        lat, lon = coords_map[key]
        if lat is not None and lon is not None:
            df.at[idx, "latitude"] = lat
            df.at[idx, "longitude"] = lon

df.to_csv(OUTPUT_CSV, index=False)
print("Saved updated CSV with latitude/longitude to", OUTPUT_CSV)
