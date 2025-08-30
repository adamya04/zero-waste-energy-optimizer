from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Inventory, Base
from datetime import datetime, timedelta
import random
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tronwaste")

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

store_ids = [f"store_{i}" for i in range(1, 11)]
items = ["Apple", "Banana", "Carrot", "Milk", "Bread", "Rice", "Chicken", "Fish", "Eggs", "Cheese"]
weather_conditions = ["Sunny", "Rainy", "Cloudy", "Stormy"]
actions = ["Keep", "Discount", "Donate"]

for _ in range(200):
    store_id = random.choice(store_ids)
    item = random.choice(items)
    date = datetime.now() - timedelta(days=random.randint(0, 90))
    expiry_date = date + timedelta(days=random.randint(1, 15))
    spoilage_rate = random.uniform(0, 1)
    stock_lbs = random.uniform(50, 1500)
    new_entry = Inventory(
        store_id=store_id,
        item=item,
        date=date,
        temperature_c=random.uniform(10, 35),
        humidity_percent=random.uniform(30, 90),
        pressure_mb=random.uniform(970, 1030),
        wind_speed_mps=random.uniform(0, 15),
        stock_lbs=stock_lbs,
        sales_lbs_daily=random.uniform(5, 150),
        expiry_date=expiry_date,
        weather=random.choice(weather_conditions),
        spoilage_rate=spoilage_rate,
        co2_emission_factor=random.uniform(0.1, 2.5),
        supply_chain_delay=random.uniform(0.5, 7),
        packaging_waste=random.uniform(0, 15),
        transport_distance_km=random.uniform(20, 1000),
        sustainability_score=random.uniform(0.2, 0.9),
        sdg_alignment=random.uniform(0.2, 0.9),
        predicted_spoilage_risk=spoilage_rate,  # Simulated as spoilage_rate for demo
        recommended_stock_lbs=stock_lbs * (1 - spoilage_rate) * 1.2,  # Simple heuristic
        action=random.choice(actions),
        tx_hash="0x" + "".join([hex(random.randint(0, 15))[2:] for _ in range(64)]) if random.random() > 0.3 else None,
        transaction_type=random.choice(["donation", "sustainability", None])
    )
    session.add(new_entry)
session.commit()
print("200 sample records inserted into inventory table for showcase.")