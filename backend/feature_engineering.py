import pandas as pd
import numpy as np

df = pd.read_csv("data/cleaned_inventory_data.csv")

# Feature engineering
df["sales_trend"] = df["sales_lbs_daily"].pct_change().fillna(0)
df["seasonality"] = np.sin(2 * np.pi * pd.to_datetime(df["date"]).dt.dayofyear / 365)
df["spoilage_risk_base"] = df["spoilage_rate"] * (df["temperature_c"] / 35) * (df["humidity_percent"] / 100)
df["environmental_impact"] = df["stock_lbs"] * df["co2_emission_factor"]
df["supply_efficiency"] = 1 / (df["supply_chain_delay"] + 1)
df["store_activity_index"] = df.groupby("store_id")["sales_lbs_daily"].transform("mean") / df["stock_lbs"]
df["weather_impact"] = df["weather"].map({"sunny": 0.1, "rainy": 0.3, "cloudy": 0.2, "humid": 0.4})
df["sustainability_score"] = (1 - df["spoilage_rate"]) * (1 - df["packaging_waste"]) * (1 / (df["transport_distance_km"] / 500))
df["sdg_alignment"] = df["sustainability_score"] * 100  # SDG proxy

df.to_csv("data/featured_inventory_data.csv", index=False)
print("Engineered features saved to data/featured_inventory_data.csv")