from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tronwaste")

engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Inventory(Base):
    __tablename__ = 'inventory'
    id = Column(Integer, primary_key=True)
    store_id = Column(String)
    item = Column(String)
    date = Column(DateTime)
    temperature_c = Column(Float)
    humidity_percent = Column(Float)
    pressure_mb = Column(Float)
    wind_speed_mps = Column(Float)
    stock_lbs = Column(Float)
    sales_lbs_daily = Column(Float)
    expiry_date = Column(DateTime)
    weather = Column(String)
    spoilage_rate = Column(Float)
    co2_emission_factor = Column(Float)
    supply_chain_delay = Column(Float)
    packaging_waste = Column(Float)
    transport_distance_km = Column(Float)
    sustainability_score = Column(Float)
    sdg_alignment = Column(Float)
    predicted_spoilage_risk = Column(Float)
    recommended_stock_lbs = Column(Float)
    action = Column(String)
    tx_hash = Column(String)
    transaction_type = Column(String)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
print("Database schema created for tronwaste database with blockchain support")