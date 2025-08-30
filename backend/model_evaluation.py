'''

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import networkx as nx
import random

# Define TransformerModel
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

# Define GRUModel to match saved architecture
class GRUModel(nn.Module):
    def __init__(self, input_dim, units1=256, units2=32, dropout_rate=0.4):
        super(GRUModel, self).__init__()
        input_dim = int(float(input_dim))
        units1 = int(float(units1))
        units2 = int(float(units2))
        dropout_rate = max(0.1, float(dropout_rate))
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

# Load and merge data with error handling
try:
    df_selected = pd.read_csv("data/selected_inventory_data.csv")
    df_featured = pd.read_csv("data/featured_inventory_data.csv")
    df = pd.merge(df_selected, df_featured[['co2_emission_factor', 'packaging_waste']], left_index=True, right_index=True, how='left')
    X = df.drop(["spoilage_risk", "store_id", "item", "date", "weather", "sustainability_score", "sdg_alignment", "store_activity_index", "supply_efficiency"], axis=1, errors='ignore')
    y = df["spoilage_risk"]
except FileNotFoundError as e:
    print(f"Error: {e}. Check data file paths.")
    exit(1)

# Debug: Print feature names to verify
print("Current feature names in X:", X.columns.tolist())

# Load scaler and align features
scaler = joblib.load("data/scaler.pkl")
print("Scaler feature names:", scaler.feature_names_in_)
original_features = ['temperature_c', 'humidity_percent', 'pressure_mb', 'wind_speed_mps', 'sales_lbs_daily', 'spoilage_rate', 'supply_chain_delay', 'transport_distance_km', 'sales_trend', 'weather_impact', 'packaging_waste_x', 'environmental_impact', 'co2_emission_factor', 'packaging_waste_y']
X = X[original_features]

# Scale the original 2D data for non-sequence models
X_scaled_2d = scaler.transform(X)

# Create sliding windows for LSTM (window_size=3)
window_size = 3
n_samples = X.shape[0] - (window_size - 1)
if n_samples <= 0:
    print("Error: Not enough samples for windowing. Check data size.")
    exit(1)
X_windowed = np.array([X.iloc[i:i+window_size].values for i in range(n_samples)])

# Scale the windowed data for sequence models
X_scaled_3d = scaler.transform(X_windowed.reshape(-1, X_windowed.shape[2])).reshape(n_samples, window_size, -1)

if "date" in df.columns:
    X_scaled_lstm = X_scaled_3d
else:
    X_scaled_lstm = X_scaled_3d.reshape((X_scaled_3d.shape[0], 1, X_scaled_3d.shape[1]))

# Load models with error handling
try:
    lstm_model = load_model("data/lstm_spoilage_model.keras")
    gru_model = GRUModel(input_dim=X_scaled_lstm.shape[2])
    gru_model.load_state_dict(torch.load("data/gru_spoilage_model.pt"))
    gru_model.eval()
    transformer_model = TransformerModel(input_dim=X_scaled_lstm.shape[2])
    transformer_model.load_state_dict(torch.load("data/transformer_spoilage_model.pt"))
    transformer_model.eval()
    xgb_model = joblib.load("data/xgboost_spoilage_model.pkl")
    rf_model = joblib.load("data/randomforest_spoilage_model.pkl")
except FileNotFoundError as e:
    print(f"Error loading models: {e}. Check model file paths.")
    exit(1)

# Evaluate models
models = {"LSTM": lstm_model, "GRU": gru_model, "Transformer": transformer_model, "XGBoost": xgb_model, "RandomForest": rf_model}
results = {}
predictions = {}

for name, model in models.items():
    try:
        if name in ["LSTM", "GRU", "Transformer"]:
            if name == "LSTM":
                y_pred = model.predict(X_scaled_lstm).flatten()
            else:
                X_torch = torch.FloatTensor(X_scaled_lstm)
                y_pred = model(X_torch).detach().numpy().flatten()
        else:
            y_pred = model.predict(X_scaled_2d[window_size-1:])  # Use 2D data for non-sequence models
        mse = mean_squared_error(y[window_size-1:], y_pred)
        r2 = r2_score(y[window_size-1:], y_pred)
        results[name] = {"MSE": mse, "R2": r2}
        predictions[name] = y_pred
    except Exception as e:
        print(f"Error evaluating {name} model: {e}")
        continue

# Ensemble Weighting
if results:
    r2_weights = {name: results[name]["R2"] for name in results}
    total_r2 = sum(r2_weights.values())
    weighted_predictions = np.zeros_like(y[window_size-1:])
    for name in predictions:
        weighted_predictions += predictions[name] * (r2_weights[name] / total_r2)
    ensemble_mse = mean_squared_error(y[window_size-1:], weighted_predictions)
    ensemble_r2 = r2_score(y[window_size-1:], weighted_predictions)
    results["Ensemble"] = {"MSE": ensemble_mse, "R2": ensemble_r2}
    predictions["Ensemble"] = weighted_predictions
else:
    print("No model evaluations completed. Skipping ensemble.")
    exit(1)

# Add predictions to df
for name in predictions:
    df[f"{name.lower()}_pred"] = np.concatenate([predictions[name], np.full(window_size-1, np.nan)])

# Enhanced Textual Analysis
# Sustainability Metrics Report
sustainability_metrics = df.groupby("store_id").agg({
    "sustainability_score": "mean",
    "co2_emission_factor": "mean",
    "sdg_alignment": "mean",
    "environmental_impact": "mean"
}).reset_index()
sustainability_metrics["rank_sustainability"] = sustainability_metrics["sustainability_score"].rank(ascending=False)
sustainability_metrics["co2_reduction_potential"] = (sustainability_metrics["co2_emission_factor"].max() - sustainability_metrics["co2_emission_factor"]) / sustainability_metrics["co2_emission_factor"].max() * 100
sustainability_metrics["recommendation"] = np.where(
    sustainability_metrics["sustainability_score"] < sustainability_metrics["sustainability_score"].mean(),
    "Implement green practices to boost sustainability score",
    "Maintain current sustainable operations"
)
sustainability_metrics.to_csv("analytics/sustainability_metrics_report.csv", index=False)

# Supply Chain Optimization Report
supply_chain_data = df.groupby("store_id").agg({
    "supply_chain_delay": "mean",
    "supply_efficiency": "mean",
    "transport_distance_km": "mean",
    "store_activity_index": "mean"
}).reset_index()
supply_chain_data["rank_efficiency"] = supply_chain_data["supply_efficiency"].rank(ascending=False)
supply_chain_data["delay_impact"] = supply_chain_data["supply_chain_delay"] / supply_chain_data["supply_chain_delay"].mean() * 100
supply_chain_data["optimization_potential"] = np.where(
    supply_chain_data["supply_chain_delay"] > supply_chain_data["supply_chain_delay"].mean(),
    (supply_chain_data["supply_chain_delay"] - supply_chain_data["supply_chain_delay"].mean()) / supply_chain_data["supply_chain_delay"].max() * 100,
    0
)
supply_chain_data["recommendation"] = np.where(
    supply_chain_data["supply_chain_delay"] > supply_chain_data["supply_chain_delay"].mean(),
    f"Reduce delay by {supply_chain_data['optimization_potential'].round(1).astype(str)}% with optimized routing",
    "Maintain current efficient supply chain"
)
supply_chain_data.to_csv("analytics/supply_chain_optimization_report.csv", index=False)

# Plot for Textual Insights
plt.figure(figsize=(14, 7), facecolor='#1A1A2E')
sns.barplot(data=supply_chain_data.sort_values("supply_efficiency", ascending=False).head(5), x="store_id", y="supply_efficiency", hue="store_id", palette=["#00FF00", "#00FFFF", "#FF00FF", "#00FFFF", "#00FF00"], legend=False)
plt.title("Top 5 Stores by Supply Efficiency", color="#00FFFF", fontsize=16, weight='bold')
plt.xlabel("Store ID", color="#00FFFF")
plt.ylabel("Supply Efficiency", color="#00FFFF")
plt.gca().set_facecolor('#16213E')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(2)
plt.savefig("analytics/top_supply_efficiency_stores.png", facecolor='#1A1A2E')
plt.close()

plt.figure(figsize=(14, 7), facecolor='#1A1A2E')
sns.barplot(data=sustainability_metrics.sort_values("sustainability_score", ascending=False).head(5), x="store_id", y="sustainability_score", hue="store_id", palette=["#00FF00", "#00FFFF", "#FF00FF", "#00FFFF", "#00FF00"], legend=False)
plt.title("Top 5 Stores by Sustainability Score", color="#00FFFF", fontsize=16, weight='bold')
plt.xlabel("Store ID", color="#00FFFF")
plt.ylabel("Sustainability Score", color="#00FFFF")
plt.gca().set_facecolor('#16213E')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(2)
plt.savefig("analytics/top_sustainability_stores.png", facecolor='#1A1A2E')
plt.close()

# Visualizations with Tron-inspired design
# 1. 3D Seasonal Sustainability Flow
df["season"] = pd.to_datetime(df["date"], errors='coerce').dt.quarter
pivot_data = df.pivot_table(values="sustainability_score", index="store_id", columns="season", aggfunc="mean").fillna(0)
stores = pivot_data.index
seasons = pivot_data.columns
x, y = np.meshgrid(np.arange(len(seasons)), np.array([int(store.split('_')[1]) for store in stores]))
z = pivot_data.values
x_2d, y_2d = np.meshgrid(seasons, [int(store.split('_')[1]) for store in stores])
fig_flow = plt.figure(figsize=(14, 7), facecolor='#1A1A2E')
ax = fig_flow.add_subplot(111, projection='3d')
ax.plot_surface(x_2d, y_2d, z, cmap='viridis', edgecolor='#00FFFF', linewidth=0.1)
ax.set_title("3D Seasonal Sustainability Flow", color="#00FFFF", fontsize=16, weight='bold')
ax.set_xlabel("Season", color="#00FFFF")
ax.set_ylabel("Store ID", color="#00FFFF")
ax.set_zlabel("Sustainability Score", color="#00FFFF")
ax.set_xticks(np.arange(len(seasons)))
ax.set_xticklabels(seasons)
ax.set_facecolor('#16213E')
for spine in ax.spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(2)
plt.savefig("analytics/3d_seasonal_sustainability_flow.png", facecolor='#1A1A2E')
plt.close()

# 2. Neon Supply Chain Delay Network
G = nx.Graph()
for store in df["store_id"].unique():
    G.add_node(store)
for i in df.index:
    for j in df.index:
        if df.loc[i, "store_id"] != df.loc[j, "store_id"]:
            G.add_edge(df.loc[i, "store_id"], df.loc[j, "store_id"], weight=df.loc[i, "supply_chain_delay"])
pos = nx.spring_layout(G)
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#00FFFF'), hoverinfo='none', mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                        marker=dict(size=10, color='#00FF00', line=dict(width=2, color='#FF00FF')))
node_trace.text = [f"Store {node}<br>Delay: {G.degree[node]:.2f}" for node in G.nodes()]

fig_network = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title="Neon Supply Chain Delay Network", titlefont_color="#00FFFF",
                                        showlegend=False, hovermode='closest',
                                        margin=dict(b=0, l=0, r=0, t=40),
                                        paper_bgcolor="rgb(10, 10, 30)", plot_bgcolor="rgb(10, 10, 30)",
                                        xaxis=dict(showgrid=False, zeroline=False),
                                        yaxis=dict(showgrid=False, zeroline=False)))
fig_network.write_html("analytics/neon_supply_chain_delay_network.html")

# 3. Sustainability vs. Efficiency Heatmap
sustainability_efficiency = df.groupby("store_id").agg({"sustainability_score": "mean", "supply_efficiency": "mean"}).reset_index()
pivot_data = sustainability_efficiency.pivot_table(index="store_id", values=["sustainability_score", "supply_efficiency"], aggfunc="mean").T
plt.figure(figsize=(14, 7), facecolor='#1A1A2E')
heatmap = sns.heatmap(pivot_data, annot=True, cmap="cool", fmt=".2f", annot_kws={"color": "#00FFFF"}, cbar_kws={'label': 'Value'})
plt.title("Sustainability vs. Efficiency Heatmap", color="#00FFFF", fontsize=16, weight='bold')
plt.gca().set_facecolor('#16213E')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(2)
cbar = heatmap.collections[0].colorbar
cbar.ax.set_facecolor('#16213E')
cbar.outline.set_edgecolor('#00FFFF')
cbar.ax.tick_params(colors='#00FFFF')
plt.savefig("analytics/sustainability_efficiency_heatmap.png", facecolor='#1A1A2E')
plt.close()

# 4. Dynamic Carbon Emission Trajectory
df["timestamp"] = pd.to_datetime(df["date"], errors='coerce').astype('int64') // 10**9  # Handle invalid dates
fig_trajectory = go.Figure()
for store in df["store_id"].unique():
    store_data = df[df["store_id"] == store].sort_values("timestamp").dropna(subset=["timestamp", "transport_distance_km", "co2_emission_factor"])
    if not store_data.empty:
        fig_trajectory.add_trace(go.Scatter3d(
            x=store_data["timestamp"],
            y=store_data["transport_distance_km"],
            z=store_data["co2_emission_factor"],
            mode='lines+markers',
            line=dict(color='#00FFFF', width=2),
            marker=dict(size=4, color=store_data["sustainability_score"], colorscale='Plasma', colorbar=dict(title="Sustainability", tickfont=dict(color="#00FFFF"))),
            name=f"Store {store}"
        ))
fig_trajectory.update_layout(
    title="Dynamic Carbon Emission Trajectory",
    title_font=dict(color="#00FFFF", size=20),
    scene=dict(xaxis_title="Timestamp", yaxis_title="Transport Distance (km)", zaxis_title="CO2 Emission Factor", xaxis=dict(color="#00FFFF"), yaxis=dict(color="#00FFFF"), zaxis=dict(color="#00FFFF")),
    paper_bgcolor="rgb(10, 10, 30)",
    plot_bgcolor="rgb(10, 10, 30)",
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
)
fig_trajectory.write_html("analytics/dynamic_carbon_emission_trajectory.html")

# 5. Weather Impact Pulse Globe
store_locs = df.groupby("store_id").agg({"weather_impact": "mean", "sustainability_score": "mean"}).reset_index()
store_locs["lat"] = [random.uniform(-90, 90) for _ in range(len(store_locs))]
store_locs["lon"] = [random.uniform(-180, 180) for _ in range(len(store_locs))]
fig_globe = go.Figure(data=go.Scattergeo(
    lon=store_locs["lon"],
    lat=store_locs["lat"],
    text=store_locs["store_id"],
    mode="markers",
    marker=dict(
        size=store_locs["weather_impact"] * 10,
        color=store_locs["sustainability_score"],
        colorscale="Viridis",
        colorbar=dict(title="Sustainability Score", tickfont=dict(color="#00FFFF")),
        line=dict(color="#FF00FF", width=1.5)
    ),
    hoverinfo="text+lat+lon"
))
fig_globe.update_layout(
    title="Weather Impact Pulse Globe",
    title_font=dict(color="#00FFFF", size=20),
    geo=dict(scope="world", projection_type="orthographic", showland=True, landcolor="rgb(20, 20, 40)", countrycolor="rgb(50, 50, 70)", showocean=True, oceancolor="rgb(10, 10, 30)"),
    paper_bgcolor="rgb(10, 10, 30)",
    plot_bgcolor="rgb(10, 10, 30)",
    height=700,
    margin=dict(l=0, r=0, t=50, b=0),
    updatemenus=[dict(type="buttons", showactive=False, y=0, x=1.1, buttons=[dict(label="Pulse", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])]
)
fig_globe.write_html("analytics/weather_impact_pulse_globe.html")

print("Evaluation results, enhanced textual analyses, and Tron-inspired visualizations saved to analytics/ directory")

'''

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import networkx as nx
import random

# Define TransformerModel
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

# Define GRUModel
class GRUModel(nn.Module):
    def __init__(self, input_dim, units1=256, units2=32, dropout_rate=0.4):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(int(float(input_dim)), int(float(units1)), batch_first=True)
        self.dropout1 = nn.Dropout(max(0.1, float(dropout_rate)))
        self.gru2 = nn.GRU(int(float(units1)), int(float(units2)))
        self.dropout2 = nn.Dropout(max(0.1, float(dropout_rate)))
        self.fc1 = nn.Linear(int(float(units2)), 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x[:, -1, :])
        x, _ = self.gru2(x.unsqueeze(1))
        x = self.dropout2(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load and merge data
try:
    df_selected = pd.read_csv("data/selected_inventory_data.csv")
    df_featured = pd.read_csv("data/featured_inventory_data.csv")
    df = pd.merge(df_selected, df_featured[['co2_emission_factor', 'packaging_waste']], left_index=True, right_index=True, how='left')
    X = df.drop(["spoilage_risk", "store_id", "item", "date", "weather", "sustainability_score", "sdg_alignment", "store_activity_index", "supply_efficiency"], axis=1, errors='ignore')
    y = df["spoilage_risk"]
except FileNotFoundError as e:
    print(f"Error: {e}. Check data file paths.")
    exit(1)

print("Current feature names in X:", X.columns.tolist())
scaler = joblib.load("data/scaler.pkl")
print("Scaler feature names:", scaler.feature_names_in_)
X = X[['temperature_c', 'humidity_percent', 'pressure_mb', 'wind_speed_mps', 'sales_lbs_daily', 'spoilage_rate', 'supply_chain_delay', 'transport_distance_km', 'sales_trend', 'weather_impact', 'packaging_waste_x', 'environmental_impact', 'co2_emission_factor', 'packaging_waste_y']]
X_scaled_2d = scaler.transform(X)
window_size = 3
n_samples = X.shape[0] - window_size + 1
print(f"n_samples: {n_samples}")  # Debug
X_windowed = np.array([X.iloc[i:i+window_size].values for i in range(n_samples)])
X_scaled_3d = scaler.transform(X_windowed.reshape(-1, X_windowed.shape[2])).reshape(n_samples, window_size, -1)
X_scaled_lstm = X_scaled_3d if "date" in df.columns else X_scaled_3d.reshape(n_samples, 1, -1)

# Load models
try:
    lstm_model = load_model("data/lstm_spoilage_model.keras")
    gru_model = GRUModel(input_dim=X_scaled_lstm.shape[2])
    gru_model.load_state_dict(torch.load("data/gru_spoilage_model.pt"))
    gru_model.eval()
    transformer_model = TransformerModel(input_dim=X_scaled_lstm.shape[2])
    transformer_model.load_state_dict(torch.load("data/transformer_spoilage_model.pt"))
    transformer_model.eval()
    xgb_model = joblib.load("data/xgboost_spoilage_model.pkl")
    rf_model = joblib.load("data/randomforest_spoilage_model.pkl")
except FileNotFoundError as e:
    print(f"Error loading models: {e}. Check model file paths.")
    exit(1)

# Evaluate models
models = {"LSTM": lstm_model, "GRU": gru_model, "Transformer": transformer_model, "XGBoost": xgb_model, "RandomForest": rf_model}
results = {}
predictions = {}

for name, model in models.items():
    try:
        if name in ["LSTM", "GRU", "Transformer"]:
            if name == "LSTM":
                y_pred = model.predict(X_scaled_lstm).flatten()
            else:
                X_torch = torch.FloatTensor(X_scaled_lstm)
                y_pred = model(X_torch).detach().numpy().flatten()
            y_pred = y_pred[:len(y[window_size-1:n_samples])]
        else:
            y_pred = model.predict(X_scaled_2d[window_size-1:n_samples])
        print(f"Prediction length for {name}: {len(y_pred)}")
        print(f"Target length for {name}: {len(y[window_size-1:n_samples])}")
        if len(y_pred) != len(y[window_size-1:n_samples]):
            raise ValueError(f"Mismatch in prediction and target lengths: {len(y_pred)} vs {len(y[window_size-1:n_samples])}")
        mse = mean_squared_error(y[window_size-1:n_samples], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y[window_size-1:n_samples], y_pred)
        results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}
        predictions[name] = y_pred
    except Exception as e:
        print(f"Error evaluating {name} model: {e}")
        continue

# Ensemble Weighting
if results:
    r2_weights = {name: results[name]["R2"] for name in results}
    total_r2 = sum(r2_weights.values())
    weighted_predictions = np.zeros_like(y[window_size-1:n_samples])
    for name in predictions:
        weighted_predictions += predictions[name] * (r2_weights[name] / total_r2)
    ensemble_mse = mean_squared_error(y[window_size-1:n_samples], weighted_predictions)
    ensemble_rmse = np.sqrt(ensemble_mse)
    ensemble_r2 = r2_score(y[window_size-1:n_samples], weighted_predictions)
    results["Ensemble"] = {"MSE": ensemble_mse, "RMSE": ensemble_rmse, "R2": ensemble_r2}
    predictions["Ensemble"] = weighted_predictions

# Save performance metrics
performance_df = pd.DataFrame(results).T
performance_df.to_csv("analytics/performance_metrics.csv")

# Add predictions to df
for name in predictions:
    df[f"{name.lower()}_pred"] = np.nan
    df.loc[window_size-1:n_samples-1, f"{name.lower()}_pred"] = predictions[name]

# Calculate overall_score on original df
df["co2_reduction_potential"] = ((df["co2_emission_factor"].max() - df["co2_emission_factor"]) / df["co2_emission_factor"].max() * 100).round(1)
df["environmental_impact_rank"] = df["environmental_impact"].rank(ascending=True)
df["overall_score"] = (df["sustainability_score"] * 0.4 + df["co2_reduction_potential"] * 0.3 + (1 - df["environmental_impact_rank"] / len(df)) * 0.3)

# Enhanced Textual Analysis
sustainability_metrics = df.groupby("store_id").agg({
    "sustainability_score": "mean",
    "co2_emission_factor": "mean",
    "sdg_alignment": "mean",
    "environmental_impact": "mean",
    "supply_efficiency": "mean",
    "overall_score": "mean"
}).reset_index()
sustainability_metrics["rank_sustainability"] = sustainability_metrics["sustainability_score"].rank(ascending=False)
sustainability_metrics["environmental_impact_rank"] = sustainability_metrics["environmental_impact"].rank(ascending=True)
sustainability_metrics["co2_reduction_potential"] = sustainability_metrics["co2_emission_factor"].max() - sustainability_metrics["co2_emission_factor"]
sustainability_metrics["recommendation"] = np.where(
    sustainability_metrics["overall_score"] < sustainability_metrics["overall_score"].mean(),
    f"Adopt energy-efficient logistics and reduce emissions by {sustainability_metrics['co2_reduction_potential'].max():.1f}%",
    "Sustain eco-friendly practices; explore advanced recycling"
)
sustainability_metrics.to_csv("analytics/sustainability_metrics_report.csv", index=False)

supply_chain_data = df.groupby("store_id").agg({
    "supply_chain_delay": "mean",
    "supply_efficiency": "mean",
    "transport_distance_km": "mean",
    "store_activity_index": "mean"
}).reset_index()
supply_chain_data["rank_efficiency"] = supply_chain_data["supply_efficiency"].rank(ascending=False)
supply_chain_data["delay_impact"] = (supply_chain_data["supply_chain_delay"] / supply_chain_data["supply_chain_delay"].mean() * 100).round(1)
supply_chain_data["optimization_potential"] = np.where(
    supply_chain_data["supply_chain_delay"] > supply_chain_data["supply_chain_delay"].mean(),
    ((supply_chain_data["supply_chain_delay"] - supply_chain_data["supply_chain_delay"].mean()) / supply_chain_data["supply_chain_delay"].max() * 100).round(1),
    0
)
supply_chain_data["recommendation"] = np.where(
    supply_chain_data["supply_chain_delay"] > supply_chain_data["supply_chain_delay"].mean(),
    f"Optimize routing to reduce delay by {supply_chain_data['optimization_potential']}%",
    "Maintain high-efficiency supply chain; monitor for upgrades"
)
supply_chain_data.to_csv("analytics/supply_chain_optimization_report.csv", index=False)

# Plots with Tron visuals
plt.figure(figsize=(12, 6), facecolor='#1A1A2E')
sns.barplot(data=supply_chain_data.sort_values("supply_efficiency", ascending=False).head(3), x="store_id", y="supply_efficiency", hue="store_id", palette=["#FF0000", "#00FF00", "#FF00FF"], legend=False)
plt.title("Top 3 Stores by Supply Efficiency", color="#00FFFF", fontsize=14)
plt.xlabel("Store ID", color="#00FFFF")
plt.ylabel("Supply Efficiency", color="#00FFFF")
plt.gca().set_facecolor('#16213E')
plt.tick_params(colors="#00FFFF")
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("analytics/top_supply_efficiency_stores.png", facecolor='#1A1A2E')
plt.close()

plt.figure(figsize=(12, 6), facecolor='#1A1A2E')
sns.barplot(data=sustainability_metrics.sort_values("overall_score", ascending=False).head(3), x="store_id", y="overall_score", hue="store_id", palette=["#FF0000", "#00FF00", "#FF00FF"], legend=False)
plt.title("Top 3 Stores by Overall Sustainability", color="#00FFFF", fontsize=14)
plt.xlabel("Store ID", color="#00FFFF")
plt.ylabel("Overall Score", color="#00FFFF")
plt.gca().set_facecolor('#16213E')
plt.tick_params(colors="#00FFFF")
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("analytics/top_sustainability_stores.png", facecolor='#1A1A2E')
plt.close()

# Time-based bar plot
df["quarter"] = pd.to_datetime(df["date"], errors='coerce').dt.quarter
time_data = df.groupby(["store_id", "quarter"]).agg({"overall_score": "mean"}).reset_index()
pivot_data = time_data.pivot(index="store_id", columns="quarter", values="overall_score").fillna(0)
print(f"pivot_data shape: {pivot_data.shape}")  # Debug
if pivot_data.empty:
    print("Warning: pivot_data is empty. Check 'date' or 'overall_score' data.")
plt.figure(figsize=(12, 6), facecolor='#1A1A2E')
pivot_data.plot(kind="bar", ax=plt.gca(), color=["#FF0000", "#00FF00", "#00FFFF", "#FF00FF"])
plt.title("Overall Sustainability Score by Quarter", color="#00FFFF", fontsize=14)
plt.xlabel("Store ID", color="#00FFFF")
plt.ylabel("Overall Score", color="#00FFFF")
plt.gca().set_facecolor('#16213E')
plt.tick_params(colors="#00FFFF")
plt.legend(title="Quarter", title_fontsize=12, labelcolor="#00FFFF")
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#00FFFF')
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("analytics/sustainability_by_quarter.png", facecolor='#1A1A2E')
plt.close()

G = nx.Graph()
G.add_nodes_from(df["store_id"].unique())
edges = [(df.loc[i, "store_id"], df.loc[j, "store_id"]) for i in df.index[::10] for j in df.index[::10] if i != j]
G.add_edges_from(edges[:50], weight=df["supply_chain_delay"].mean())
pos = nx.spring_layout(G, seed=42)
edge_x = []
edge_y = []
for edge in edges[:50]:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#00FFFF'), hoverinfo='none', mode='lines')
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                        marker=dict(size=10, color='#00FF00', line=dict(width=1.5, color='#FF00FF')),
                        text=[f"Store {node}<br>Degree: {G.degree[node]:.1f}<br>Avg Delay: {df[df['store_id'] == node]['supply_chain_delay'].mean():.1f}" for node in G.nodes()])
fig_network = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title="Supply Chain Delay Network (Hover for Store Details)", title_font_color="#00FFFF",
                                         showlegend=False, hovermode='closest',
                                         margin=dict(b=0, l=0, r=0, t=40),
                                         paper_bgcolor="rgb(10, 10, 30)", plot_bgcolor="rgb(10, 10, 30)",
                                         xaxis=dict(showgrid=False, zeroline=False),
                                         yaxis=dict(showgrid=False, zeroline=False)))
fig_network.write_html("analytics/supply_chain_delay_network.html")

df["timestamp"] = pd.to_datetime(df["date"], errors='coerce').astype('int64') // 10**9
fig_trajectory = go.Figure()
for store in df["store_id"].unique()[:5]:
    store_data = df[df["store_id"] == store].sort_values("timestamp").dropna(subset=["timestamp", "transport_distance_km", "co2_emission_factor"]).iloc[::2]
    if not store_data.empty:
        fig_trajectory.add_trace(go.Scatter3d(
            x=store_data["timestamp"],
            y=store_data["transport_distance_km"],
            z=store_data["co2_emission_factor"],
            mode='lines+markers',
            line=dict(color='#00FFFF', width=1.5),
            marker=dict(size=3, color=store_data["sustainability_score"], colorscale='Plasma', colorbar=dict(title="Sustainability", tickfont=dict(color="#00FFFF"))),
            name=f"Store {store}"
        ))
fig_trajectory.update_layout(
    title="Carbon Emission Trajectory", title_font=dict(color="#00FFFF", size=16),
    scene=dict(xaxis_title="Timestamp", yaxis_title="Distance (km)", zaxis_title="CO2 Factor", xaxis=dict(color="#00FFFF"), yaxis=dict(color="#00FFFF"), zaxis=dict(color="#00FFFF")),
    paper_bgcolor="rgb(10, 10, 30)", plot_bgcolor="rgb(10, 10, 30)",
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
)
fig_trajectory.write_html("analytics/carbon_emission_trajectory.html")

store_locs = df.groupby("store_id").agg({"weather_impact": "mean", "sustainability_score": "mean"}).reset_index().sample(frac=0.5)
store_locs["lat"] = [random.uniform(-90, 90) for _ in range(len(store_locs))]
store_locs["lon"] = [random.uniform(-180, 180) for _ in range(len(store_locs))]
fig_globe = go.Figure(data=go.Scattergeo(
    lon=store_locs["lon"], lat=store_locs["lat"], text=store_locs["store_id"],
    mode="markers", marker=dict(size=store_locs["weather_impact"] * 20 + 10, color=store_locs["sustainability_score"],
                                colorscale="Viridis", colorbar=dict(title="Sustainability", tickfont=dict(color="#00FFFF")),
                                line=dict(color="#FF00FF", width=1))
))
fig_globe.update_layout(
    title="Weather Impact Globe", title_font=dict(color="#00FFFF", size=16),
    geo=dict(scope="world", projection_type="orthographic", showland=True, landcolor="rgb(20, 20, 40)",
             countrycolor="rgb(50, 50, 70)", showocean=True, oceancolor="rgb(10, 10, 30)"),
    paper_bgcolor="rgb(10, 10, 30)", plot_bgcolor="rgb(10, 10, 30)", height=600,
    margin=dict(l=0, r=0, t=40, b=0)
)
fig_globe.write_html("analytics/weather_impact_globe.html")

print("Results, metrics, and visualizations saved to analytics/ directory")