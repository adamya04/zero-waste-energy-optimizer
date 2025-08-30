import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Numeric columns from Inventory schema
NUMERIC_COLUMNS = [
    'temperature_c', 'humidity_percent', 'pressure_mb', 'wind_speed_mps',
    'stock_lbs', 'sales_lbs_daily', 'spoilage_rate', 'co2_emission_factor',
    'supply_chain_delay', 'packaging_waste', 'transport_distance_km'
]

# Load data
try:
    engine = create_engine(os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tronwaste"))
    df = pd.read_sql("SELECT * FROM inventory", engine)
except Exception as e:
    logging.error(f"Error loading data: {str(e)}")
    raise Exception(f"Error loading data: {str(e)}")

# Check and convert date
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dtype != "datetime64[ns]":
            raise ValueError("Date conversion failed")
        logging.info("Date column converted to datetime")
    except ValueError as e:
        logging.warning(f"Error converting date: {e}. Using as-is.")
        df["date"] = pd.to_datetime(df["date"], errors="ignore")

# Prepare features
available_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
if 'spoilage_rate' not in df.columns:
    logging.error("spoilage_rate column missing from inventory table")
    raise Exception("spoilage_rate column missing from inventory table")
X = df[available_columns].fillna(0)
y = df["spoilage_rate"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "data/scaler.pkl")

# Time-series preparation
window_size = 3
if "date" in df.columns and df["date"].dtype == "datetime64[ns]":
    df = df.sort_values("date")
    X_time_series = []
    y_time_series = []
    for i in range(len(df) - window_size + 1):
        X_time_series.append(X_scaled[i:i + window_size])
        y_time_series.append(y.iloc[i + window_size - 1])
    X_scaled_lstm = np.array(X_time_series)
    y = np.array(y_time_series)
    # Trim X_scaled for non-time-series models to match y
    X_scaled = X_scaled[window_size - 1:] if len(X_scaled) >= window_size else X_scaled
else:
    X_scaled_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    window_size = 1

# Verify sample sizes
if len(X_scaled) != len(y):
    logging.error(f"Inconsistent sample sizes: X_scaled={len(X_scaled)}, y={len(y)}")
    raise ValueError(f"Inconsistent sample sizes: X_scaled={len(X_scaled)}, y={len(y)}")

# Split data
try:
    X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X_scaled_lstm, y, test_size=0.2, random_state=42)
    X_train_non_lstm, X_test_non_lstm, y_train_non_lstm, y_test_non_lstm = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
except Exception as e:
    logging.error(f"Error splitting data: {str(e)}")
    raise Exception(f"Error splitting data: {str(e)}")

# LSTM Model
def build_lstm_model(timesteps, features):
    inputs = Input(shape=(timesteps, features))
    x = LSTM(128, return_sequences=True, kernel_regularizer='l2')(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64, kernel_regularizer='l2')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.0005), loss="mse")
    try:
        model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=0)
        model.save("data/lstm_spoilage_model.keras")
    except Exception as e:
        logging.error(f"Error training LSTM: {str(e)}")
        raise Exception(f"Error training LSTM: {str(e)}")
    return model

lstm_model = build_lstm_model(window_size, X_scaled.shape[1])

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_dim):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_dim, 256, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)
        self.gru2 = nn.GRU(256, 32)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x[:, -1, :])
        x, _ = self.gru2(x.unsqueeze(1))
        x = self.dropout2(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Train GRU
try:
    gru_model = GRUModel(input_dim=X_scaled_lstm.shape[2])
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    X_train_torch = torch.FloatTensor(X_train_lstm)
    y_train_torch = torch.FloatTensor(y_train.reshape(-1, 1))
    for epoch in range(50):
        gru_model.train()
        outputs = gru_model(X_train_torch)
        loss = loss_fn(outputs, y_train_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(gru_model.state_dict(), "data/gru_spoilage_model.pt")
except Exception as e:
    logging.error(f"Error training GRU: {str(e)}")
    raise Exception(f"Error training GRU: {str(e)}")

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(int(float(input_dim)), 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# Train Transformer
try:
    transformer_model = TransformerModel(input_dim=X_scaled_lstm.shape[2])
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0005)
    X_train_torch = torch.FloatTensor(X_train_lstm)
    y_train_torch = torch.FloatTensor(y_train.reshape(-1, 1))
    for epoch in range(50):
        transformer_model.train()
        outputs = transformer_model(X_train_torch)
        loss = nn.MSELoss()(outputs, y_train_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(transformer_model.state_dict(), "data/transformer_spoilage_model.pt")
except Exception as e:
    logging.error(f"Error training Transformer: {str(e)}")
    raise Exception(f"Error training Transformer: {str(e)}")

# XGBoost and Random Forest
try:
    xgb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=8)
    xgb_model.fit(X_train_non_lstm, y_train_non_lstm)
    joblib.dump(xgb_model, "data/xgboost_spoilage_model.pkl")

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf_model.fit(X_train_non_lstm, y_train_non_lstm)
    joblib.dump(rf_model, "data/randomforest_spoilage_model.pkl")
except Exception as e:
    logging.error(f"Error training XGBoost/RF: {str(e)}")
    raise Exception(f"Error training XGBoost/RF: {str(e)}")

logging.info("Generated scaler.pkl and models successfully")