import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_inventory_data.csv")

# Handle missing values and outliers
df = df.dropna()
df = df[(df["temperature_c"] <= 50) & (df["temperature_c"] >= -10)]
df = df[df["humidity_percent"] <= 100]
df["sales_lbs_daily"] = df["sales_lbs_daily"].clip(lower=0)
df["stock_lbs"] = df["stock_lbs"].clip(lower=0)
df["days_until_expiry"] = (pd.to_datetime(df["expiry_date"]) - pd.to_datetime(df["date"])).dt.days.clip(lower=0)
df["pressure_mb"] = df["pressure_mb"].clip(lower=900, upper=1100)
df["wind_speed_mps"] = df["wind_speed_mps"].clip(lower=0, upper=20)

df.to_csv("data/cleaned_inventory_data.csv", index=False)
print("Cleaned data saved to data/cleaned_inventory_data.csv")