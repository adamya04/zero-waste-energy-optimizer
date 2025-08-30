import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import random
from tensorflow.keras.models import load_model, save_model

# Load and merge data
df_selected = pd.read_csv("data/selected_inventory_data.csv")
df_featured = pd.read_csv("data/featured_inventory_data.csv")
print("Selected columns:", df_selected.columns.tolist())
print("Featured columns:", df_featured.columns.tolist())
df = pd.merge(df_selected, df_featured[['co2_emission_factor', 'packaging_waste']], left_index=True, right_index=True, how='left')
X = df.drop(["spoilage_risk", "store_id", "item", "date", "weather", "sustainability_score", "sdg_alignment", "store_activity_index", "supply_efficiency"], axis=1)
y = df["spoilage_risk"]

# Check and convert date column to datetime
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dtype != "datetime64[ns]":
            raise ValueError("Date conversion failed or resulted in non-datetime type")
        print("Date column successfully converted to datetime")
    except ValueError as e:
        print(f"Error converting date column to datetime: {e}. Using as-is or falling back to non-time-series.")
        df["date"] = pd.to_datetime(df["date"], errors="ignore")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-Series Preparation: Create sliding window if date exists and is datetime
if "date" in df.columns and df["date"].dtype == "datetime64[ns]":
    df = df.sort_values("date")
    window_size = 3  # 3 timesteps
    X_time_series = []
    y_time_series = []
    for i in range(len(df) - window_size + 1):
        X_time_series.append(X_scaled[i:i + window_size])
        y_time_series.append(y.iloc[i + window_size - 1])
    X_scaled_lstm = np.array(X_time_series)
    y = np.array(y_time_series)
    timesteps = window_size
    features = X_scaled.shape[1]
else:
    X_scaled_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    timesteps = 1
    features = X_scaled.shape[1]

# Split data
if "date" in df.columns and df["date"].dtype == "datetime64[ns]":
    X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X_scaled_lstm, y, test_size=0.2, random_state=42)
    X_train_non_lstm, X_test_non_lstm, y_train_non_lstm, y_test_non_lstm = train_test_split(
        X_scaled[:len(y)], y, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train_lstm, X_test_lstm, _, _ = train_test_split(X_scaled_lstm, y, test_size=0.2, random_state=42)

# Custom model builder for LSTM with enhanced capacity
def build_lstm_model(units1=128, units2=64, dropout_rate=0.3, learning_rate=0.0005, timesteps=None, features=None):
    inputs = Input(shape=(timesteps, features))
    x = LSTM(units1, return_sequences=True, kernel_regularizer='l2')(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units2, kernel_regularizer='l2')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

# Hyperparameter Tuning for LSTM 
best_score = float("inf")
best_params = None
best_model = None
for _ in range(10):
    params = {k: int(np.random.choice(v)) for k, v in param_dist.items() if k in ["units1", "units2"]}
    params["dropout_rate"] = float(np.random.choice(param_dist["dropout_rate"]))
    params["learning_rate"] = float(np.random.choice(param_dist["learning_rate"]))
    params["dropout_rate"] = max(0.1, params["dropout_rate"])
    params["learning_rate"] = max(0.0001, params["learning_rate"])
    model = train_lstm_model(params)
    val_loss = model.evaluate(X_test_lstm, y_test, verbose=0)
    if val_loss < best_score:
        best_score = val_loss
        best_params = params
        best_model = model
best_lstm_model = best_model
best_lstm_model.save("data/lstm_spoilage_model.keras")
print(f"Best LSTM params: {best_params}, Best Validation Loss: {best_score}")


# Define GRU Model for Hyperparameter Tuning
class GRUModel(nn.Module):
    def __init__(self, input_dim, units1=256, units2=32, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        input_dim = int(float(input_dim))
        units1 = int(float(units1))
        units2 = int(float(units2))
        dropout_rate = max(0.1, float(dropout_rate))
        print(f"GRU init - input_dim: {input_dim} (type: {type(input_dim)}), units1: {units1} (type: {type(units1)}), dropout_rate: {dropout_rate} (type: {type(dropout_rate)})")
        self.gru1 = nn.GRU(input_dim, units1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.gru2 = nn.GRU(units1, units2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(units2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x[:, -1, :])
        x, _ = self.gru2(x.unsqueeze(1))
        x = self.dropout2(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Hyperparameter Tuning for GRU 
param_dist_gru = {
    "units1": [64, 128, 256],
    "units2": [32, 64, 128],
    "dropout_rate": [0.2, 0.3, 0.4],
    "learning_rate": [0.001, 0.0005, 0.0001]
}
best_gru_loss = float("inf")
best_gru_model = None
for _ in range(10):
    params = {k: float(np.random.choice(v)) for k, v in param_dist_gru.items()}
    params["dropout_rate"] = max(0.1, params["dropout_rate"])
    params["learning_rate"] = max(0.0001, params["learning_rate"])
    print(f"GRU params: {params}")
    model = train_gru_model(**params)
    X_train_torch = torch.FloatTensor(X_train_lstm)
    y_train_torch = torch.FloatTensor(y_train.reshape(-1, 1))
    model.eval()
    with torch.no_grad():
        outputs = model(X_train_torch)
        loss = nn.MSELoss()(outputs, y_train_torch)
    if loss.item() < best_gru_loss:
        best_gru_loss = loss.item()
        best_gru_model = model
torch.save(best_gru_model.state_dict(), "data/gru_spoilage_model.pt")
print(f"Best GRU params: {best_params}")


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(int(float(input_dim)), int(float(d_model)))
        encoder_layer = nn.TransformerEncoderLayer(d_model=int(float(d_model)), nhead=int(float(nhead)), batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(float(num_layers)))
        self.fc = nn.Linear(int(float(d_model)), 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# Transformer training (commented out to skip training)

transformer_model = TransformerModel(input_dim=int(float(X_train_lstm.shape[2])))
transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0005)
X_train_torch = torch.FloatTensor(X_train_lstm)
y_train_torch = torch.FloatTensor(y_train.reshape(-1, 1))
transformer_model.train()
for epoch in range(50):
    outputs = transformer_model(X_train_torch)
    loss = nn.MSELoss()(outputs, y_train_torch)
    transformer_optimizer.zero_grad()
    loss.backward()
    transformer_optimizer.step()
torch.save(transformer_model.state_dict(), "data/transformer_spoilage_model.pt")


# XGBoost Model (commented out to skip training)

xgb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=8)
xgb_model.fit(X_train_non_lstm, y_train_non_lstm)
joblib.dump(xgb_model, "data/xgboost_spoilage_model.pkl")


# Random Forest Model (commented out to skip training)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train_non_lstm, y_train_non_lstm)
joblib.dump(rf_model, "data/randomforest_spoilage_model.pkl")


# Save scaler
joblib.dump(scaler, "data/scaler.pkl")

# Load pre-trained models with correct architecture
lstm_model = load_model("data/lstm_spoilage_model.keras")

# Define GRUModel with best parameters from training
class GRUModel(nn.Module):
    def __init__(self, input_dim, units1=256, units2=32, dropout_rate=0.4):  # Updated to match saved model
        super(GRUModel, self).__init__()
        input_dim = int(float(input_dim))
        units1 = int(float(units1))
        units2 = int(float(units2))
        dropout_rate = max(0.1, float(dropout_rate))
        print(f"GRU init - input_dim: {input_dim} (type: {type(input_dim)}), units1: {units1} (type: {type(units1)}), dropout_rate: {dropout_rate} (type: {type(dropout_rate)})")
        self.gru1 = nn.GRU(input_dim, units1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.gru2 = nn.GRU(units1, units2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(units2, 16)  # Matches [16, 32] from checkpoint
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x[:, -1, :])
        x, _ = self.gru2(x.unsqueeze(1))
        x = self.dropout2(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

gru_model = GRUModel(input_dim=X_scaled_lstm.shape[2])
gru_model.load_state_dict(torch.load("data/gru_spoilage_model.pt"))
gru_model.eval()

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(int(float(input_dim)), int(float(d_model)))
        encoder_layer = nn.TransformerEncoderLayer(d_model=int(float(d_model)), nhead=int(float(nhead)), batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(float(num_layers)))
        self.fc = nn.Linear(int(float(d_model)), 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

transformer_model = TransformerModel(input_dim=X_scaled_lstm.shape[2])
transformer_model.load_state_dict(torch.load("data/transformer_spoilage_model.pt"))
transformer_model.eval()

# Load pre-trained XGBoost and Random Forest models
xgb_model = joblib.load("data/xgboost_spoilage_model.pkl")
rf_model = joblib.load("data/randomforest_spoilage_model.pkl")

# Predictions for visualizations
lstm_pred = lstm_model.predict(X_scaled_lstm).flatten()
gru_pred = gru_model(torch.FloatTensor(X_scaled_lstm)).detach().numpy().flatten()
transformer_pred = transformer_model(torch.FloatTensor(X_scaled_lstm)).detach().numpy().flatten()
xgb_pred = xgb_model.predict(X_scaled[:len(y)])
rf_pred = rf_model.predict(X_scaled[:len(y)])
df["predicted_spoilage_risk_lstm"] = np.concatenate([lstm_pred, np.full(len(df) - len(lstm_pred), np.nan)])
df["predicted_spoilage_risk_gru"] = np.concatenate([gru_pred, np.full(len(df) - len(gru_pred), np.nan)])
df["predicted_spoilage_risk_transformer"] = np.concatenate([transformer_pred, np.full(len(df) - len(transformer_pred), np.nan)])
df["predicted_spoilage_risk_xgb"] = np.concatenate([xgb_pred, np.full(len(df) - len(xgb_pred), np.nan)])
df["predicted_spoilage_risk_rf"] = np.concatenate([rf_pred, np.full(len(df) - len(rf_pred), np.nan)])

# Visualizations with Tron-inspired design
available_cols = [col for col in ["temperature_c", "humidity_percent", "sales_lbs_daily", "pressure_mb", "wind_speed_mps", "packaging_waste", "co2_emission_factor", "transport_distance_km"] if col in df.columns]
plt.figure(figsize=(14, 7))
sns.boxplot(data=df[available_cols], palette="Blues")
plt.title("Distribution of Key Factors", color="#00FFFF", fontsize=16, weight='bold')
plt.xlabel("Factors", color="#00FFFF")
plt.ylabel("Values", color="#00FFFF")
plt.gcf().set_facecolor('#1A1A2E')
plt.gca().set_facecolor('#16213E')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(2)
plt.savefig("analytics/feature_distribution.png", facecolor='#1A1A2E')
plt.close()

# Animated 3D Trajectory of Sustainability
fig_trajectory = go.Figure()
for store_id in df["store_id"].unique():
    store_data = df[df["store_id"] == store_id].sort_values("date")
    fig_trajectory.add_trace(go.Scatter3d(
        x=store_data["date"].astype('int64') // 10**9,
        y=store_data["sustainability_score"],
        z=store_data["predicted_spoilage_risk_lstm"],
        mode='lines+markers',
        line=dict(color='#00FFFF', width=2),
        marker=dict(size=4, color=store_data["co2_emission_factor"], colorscale='Plasma', colorbar=dict(title="CO2", tickfont=dict(color="#00FFFF"))),
        name=f"Store {store_id}"
    ))
fig_trajectory.update_layout(
    title="Animated 3D Trajectory of Sustainability",
    title_font=dict(color="#00FFFF", size=20),
    scene=dict(xaxis_title="Date", yaxis_title="Sustainability Score", zaxis_title="Predicted Spoilage Risk", xaxis=dict(color="#00FFFF"), yaxis=dict(color="#00FFFF"), zaxis=dict(color="#00FFFF")),
    updatemenus=[dict(type="buttons", showactive=False, y=0, x=1.1, buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode="immediate")])])],
    paper_bgcolor="rgb(10, 10, 30)",
    plot_bgcolor="rgb(10, 10, 30)",
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
)
fig_trajectory.write_html("analytics/sustainability_trajectory.html")

# Rotating 3D Bar Chart for Waste Reduction
df["waste_reduction"] = 1 - df["predicted_spoilage_risk_lstm"]
waste_by_store = df.groupby("store_id")["waste_reduction"].mean() * 100
fig_bars = go.Figure(data=[go.Bar(x=waste_by_store.index, y=waste_by_store.values, marker_color='#00FF00')])
fig_bars.update_layout(
    title="Rotating 3D Waste Reduction by Store",
    title_font=dict(color="#00FFFF", size=20),
    xaxis_title="Store ID",
    yaxis_title="Waste Reduction (%)",
    xaxis=dict(color="#00FFFF"),
    yaxis=dict(color="#00FFFF"),
    paper_bgcolor="rgb(10, 10, 30)",
    plot_bgcolor="rgb(10, 10, 30)",
    updatemenus=[dict(type="buttons", showactive=False, y=0, x=1.1, buttons=[dict(label="Rotate", method="relayout", args=[{"scene.camera.eye": {"x": 1.5, "y": -1.5, "z": 0.5}}])])]
)
fig_bars.write_html("analytics/waste_reduction_bars.html")

# Correlation Heatmap with Enhanced Styling
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(14, 7), facecolor='#1A1A2E')
heatmap = sns.heatmap(numeric_df.corr(), annot=True, cmap="cool", fmt=".2f", annot_kws={"color": "#00FFFF"}, cbar_kws={'label': 'Correlation'})
plt.title("Correlation Heatmap of Numeric Features", color="#00FFFF", fontsize=16, weight='bold')
plt.gca().set_facecolor('#16213E')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(2)
cbar = heatmap.collections[0].colorbar
cbar.ax.set_facecolor('#16213E')
cbar.outline.set_edgecolor('#00FFFF')
cbar.ax.tick_params(colors='#00FFFF')
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels([f"{x:.1f}" for x in [-1, 0, 1]])
plt.savefig("analytics/correlation_heatmap.png", facecolor='#1A1A2E')
plt.close()

# Interactive Globe with Pulsing Effect
store_locs = df.groupby("store_id").agg({"sustainability_score": "mean", "co2_emission_factor": "mean", "predicted_spoilage_risk_lstm": "mean"}).reset_index()
store_locs["lat"] = [random.uniform(-90, 90) for _ in range(len(store_locs))]
store_locs["lon"] = [random.uniform(-180, 180) for _ in range(len(store_locs))]
fig_globe = go.Figure(data=go.Scattergeo(
    lon=store_locs["lon"],
    lat=store_locs["lat"],
    text=store_locs["store_id"],
    mode="markers",
    marker=dict(
        size=store_locs["sustainability_score"] * 15,
        color=store_locs["predicted_spoilage_risk_lstm"],
        colorscale="Viridis",
        colorbar=dict(title="Spoilage Risk", tickfont=dict(color="#00FFFF")),
        line=dict(color="#FF00FF", width=1.5)
    ),
    hoverinfo="text+lat+lon"
))
fig_globe.update_layout(
    title="Interactive Pulsing Globe of Sustainability",
    title_font=dict(color="#00FFFF", size=20),
    geo=dict(scope="world", projection_type="orthographic", showland=True, landcolor="rgb(20, 20, 40)", countrycolor="rgb(50, 50, 70)", showocean=True, oceancolor="rgb(10, 10, 30)"),
    paper_bgcolor="rgb(10, 10, 30)",
    plot_bgcolor="rgb(10, 10, 30)",
    height=700,
    margin=dict(l=0, r=0, t=50, b=0),
    updatemenus=[dict(type="buttons", showactive=False, y=0, x=1.1, buttons=[dict(label="Pulse", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])]
)
fig_globe.write_html("analytics/global_sustainability_pulse.html")

print("Models loaded and visualizations saved to data/ and analytics/ directories")