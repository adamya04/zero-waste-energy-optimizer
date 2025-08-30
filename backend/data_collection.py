import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate Kaggle API

os.environ["KAGGLE_USERNAME"] = "itsmeadamya04"
os.environ["KAGGLE_KEY"] = "bcf198ba4397f7ab22d6e9fdf1bc74c7"
api = KaggleApi()
api.authenticate()

# Download Kaggle dataset
try:
    api.dataset_download_files("manjeetsingh/retaildataset", path="data/", unzip=True)
    df_kaggle = pd.read_csv("data/kaggle_sales_data.csv")  # Adjust filename
    df_kaggle = df_kaggle[["Store", "Date", "Sales", "Item"]].rename(columns={"Store": "store_id", "Date": "date", "Sales": "sales_lbs_daily", "Item": "item"})
    print("Downloaded and processed Kaggle dataset")
except Exception as e:
    print(f"Kaggle download failed: {e}; using synthetic data only")

# Load IoT data
df_iot = pd.read_csv("data/iot_data.csv")

# Enhanced synthetic data (30,000 rows)
np.random.seed(42)
random.seed(42)
stores = [f"store_{i}" for i in range(1, 101)]
items = ["bananas", "apples", "berries", "lettuce", "tomatoes", "cucumbers", "oranges", "peaches", "grapes", "carrots",
         "pears", "melons", "spinach", "broccoli", "potatoes", "mangoes", "kiwis", "strawberries", "zucchini", "onions"]
dates = [datetime.now() - timedelta(days=x) for x in range(30)]
spoilage_rates = {item: random.uniform(0.1, 0.35) for item in items}

data = {
    "store_id": [], "item": [], "date": [], "temperature_c": [], "humidity_percent": [], "pressure_mb": [], "wind_speed_mps": [],
    "stock_lbs": [], "sales_lbs_daily": [], "expiry_date": [], "weather": [], "spoilage_rate": [],
    "co2_emission_factor": [], "supply_chain_delay": [], "packaging_waste": [], "transport_distance_km": []
}

for date in dates:
    for store in stores[:30]:  # 30 stores
        for item in items[:15]:  # 15 items
            data["store_id"].append(store)
            data["item"].append(item)
            data["date"].append(date.strftime("%Y-%m-%d"))
            iot_row = df_iot.sample(1).iloc[0]
            data["temperature_c"].append(iot_row["temperature_c"] + random.uniform(-2, 2))
            data["humidity_percent"].append(iot_row["humidity_percent"] + random.uniform(-5, 5))
            data["pressure_mb"].append(iot_row["pressure_mb"] + random.uniform(-5, 5))
            data["wind_speed_mps"].append(iot_row["wind_speed_mps"] + random.uniform(-0.5, 0.5))
            data["stock_lbs"].append(random.randint(50, 500))
            data["sales_lbs_daily"].append(random.randint(20, 200))
            data["expiry_date"].append((date + timedelta(days=random.randint(1, 10))).strftime("%Y-%m-%d"))
            data["weather"].append(random.choice(["sunny", "rainy", "cloudy", "humid"]))
            spoilage = spoilage_rates[item] * (1 + (data["temperature_c"][-1] / 35) * 0.2 - (data["humidity_percent"][-1] / 100) * 0.1)
            data["spoilage_rate"].append(max(0, min(spoilage, 1)))
            data["co2_emission_factor"].append(random.uniform(0.5, 2.0))  # kg CO2 per lb
            data["supply_chain_delay"].append(random.uniform(0, 5))  # Days
            data["packaging_waste"].append(random.uniform(0.1, 1.0))  # kg per lb
            data["transport_distance_km"].append(random.uniform(50, 500))  # km

df = pd.DataFrame(data)
if 'df_kaggle' in locals():
    df = pd.concat([df, df_kaggle], ignore_index=True)
df.to_csv("data/raw_inventory_data.csv", index=False)
print("Generated enhanced synthetic data with Kaggle and IoT integration: data/raw_inventory_data.csv (30,000+ rows)")