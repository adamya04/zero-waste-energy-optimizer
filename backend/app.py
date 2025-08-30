 
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import torch
import torch.nn as nn
from sqlalchemy.orm import sessionmaker
from database_setup import Inventory, engine
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import datetime
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.io as pio
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

app = Flask(__name__, static_folder='analytics')
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/analytics/*": {"origins": "*"}, r"/socket.io/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Define numeric columns from Inventory schema
NUMERIC_COLUMNS = [
    'temperature_c', 'humidity_percent', 'pressure_mb', 'wind_speed_mps',
    'stock_lbs', 'sales_lbs_daily', 'spoilage_rate', 'co2_emission_factor',
    'supply_chain_delay', 'packaging_waste', 'transport_distance_km'
]

# Load data from database
try:
    Session = sessionmaker(bind=engine)
    session = Session()
    df = pd.read_sql("SELECT * FROM inventory", engine)
    session.close()
    
    # Ensure only expected numeric columns are used
    available_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
    missing_columns = [col for col in NUMERIC_COLUMNS if col not in df.columns]
    if missing_columns:
        logging.warning(f"Missing columns in database: {missing_columns}. Filling with zeros.")
        for col in missing_columns:
            df[col] = 0.0
    
    X = df[available_columns]
    scaler = joblib.load("data/scaler.pkl")
    X_scaled = scaler.transform(X)
except Exception as e:
    logging.error(f"Error loading or scaling data: {str(e)}")
    raise Exception(f"Error loading or scaling data: {str(e)}")

# Time-Series Preparation
window_size = 3
n_samples = X_scaled.shape[0] - window_size + 1
if n_samples > 0:
    X_windowed = np.array([X_scaled[i:i + window_size] for i in range(n_samples)])
    X_scaled_lstm = X_windowed.reshape(n_samples, window_size, -1)
else:
    X_scaled_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    window_size = 1

# Define GRU Model
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

# Define Transformer Model
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
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    raise Exception(f"Error loading models: {str(e)}")

Session = sessionmaker(bind=engine)

@app.route('/analytics/<path:filename>')
def serve_analytics(filename):
    try:
        return send_from_directory('analytics', filename)
    except Exception as e:
        logging.error(f"Error serving analytics file: {str(e)}")
        return jsonify({"error": str(e)}), 404

@app.route("/api/inventory/<store_id>", methods=["GET"])
def get_inventory(store_id):
    try:
        session = Session()
        df_store = pd.read_sql(f"SELECT * FROM inventory WHERE store_id = '{store_id}'", engine)
        session.close()
        socketio.emit('inventory_update', df_store.to_dict(orient='records'))
        return jsonify(df_store.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"Error fetching inventory: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analytics/<store_id>", methods=["GET"])
def get_analytics(store_id):
    try:
        session = Session()
        df_store = pd.read_sql(f"SELECT * FROM inventory WHERE store_id = '{store_id}'", engine)
        available_columns = [col for col in NUMERIC_COLUMNS if col in df_store.columns]
        X_store = df_store[available_columns]
        X_store_scaled = scaler.transform(X_store)

        n_samples_store = X_store_scaled.shape[0] - window_size + 1
        if n_samples_store > 0:
            X_store_windowed = np.array([X_store_scaled[i:i + window_size] for i in range(n_samples_store)])
            X_store_scaled_lstm = X_store_windowed.reshape(n_samples_store, window_size, -1)
        else:
            X_store_scaled_lstm = X_store_scaled.reshape(X_store_scaled.shape[0], 1, X_store_scaled.shape[1])

        lstm_pred = lstm_model.predict(X_store_scaled_lstm, verbose=0).flatten()
        gru_pred = gru_model(torch.FloatTensor(X_store_scaled_lstm)).detach().numpy().flatten()
        transformer_pred = transformer_model(torch.FloatTensor(X_store_scaled_lstm)).detach().numpy().flatten()
        xgb_pred = xgb_model.predict(X_store_scaled[max(0, window_size-1):])
        rf_pred = rf_model.predict(X_store_scaled[max(0, window_size-1):])
        final_pred = (lstm_pred + gru_pred + transformer_pred + xgb_pred + rf_pred) / 5

        waste_reduction = df_store["sustainability_score"].mean() * 100 if not df_store.empty else 0
        cost_savings = (1 - df_store["spoilage_rate"].mean()) * 50 if not df_store.empty else 0
        donation_potential = df_store[df_store["expiry_date"] <= datetime.datetime.now() + datetime.timedelta(days=1)]["stock_lbs"].sum() if not df_store.empty else 0

        analytics = {
            "records": df_store.to_dict(orient="records"),
            "predicted_spoilage_risk": final_pred.tolist()[:len(df_store) - window_size + 1],
            "waste_reduction_trend": waste_reduction,
            "cost_savings": cost_savings,
            "donation_potential": donation_potential,
            "sustainability_score_avg": df_store["sustainability_score"].mean() if not df_store.empty else 0,
            "sdg_alignment_avg": df_store["sdg_alignment"].mean() if not df_store.empty else 0,
            "co2_reduction": donation_potential * df_store["co2_emission_factor"].mean() if not df_store.empty else 0
        }
        session.close()
        socketio.emit('analytics_update', {'store_id': store_id})
        return jsonify(analytics)
    except Exception as e:
        logging.error(f"Error generating analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/blockchain/transactions/<store_id>", methods=["GET"])
def get_blockchain_transactions(store_id):
    try:
        response = requests.get(f"http://localhost:5002/api/blockchain/transactions/{store_id}", timeout=10)
        response.raise_for_status()
        data = response.json()
        normalized = []
        for tx in data:
            normalized.append({
                "store_id": tx.get("store_id"),
                "item": tx.get("item"),
                "stock_lbs": tx.get("amount_lbs", tx.get("stock_lbs", 0)),
                "transaction_type": tx.get("type") or tx.get("transaction_type"),
                "date": tx.get("timestamp") or tx.get("date"),
                "tx_hash": tx.get("tx_hash"),
                "co2_emission_factor": tx.get("co2_emission_factor", 0)
            })
        socketio.emit('analytics_update', {'store_id': store_id})
        return jsonify(normalized)
    except Exception as e:
        logging.error(f"Error fetching blockchain transactions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/blockchain/analytics/<store_id>", methods=["GET"])
def get_blockchain_analytics(store_id):
    try:
        session = Session()
        df = pd.read_sql(f"SELECT * FROM inventory WHERE store_id = '{store_id}' AND tx_hash IS NOT NULL", engine)
        tx_counts = df.groupby('transaction_type').size().to_dict()
        tx_amounts = df.groupby('transaction_type').agg({'stock_lbs': 'sum', 'sustainability_score': 'sum', 'co2_emission_factor': 'mean'}).to_dict()
        
        # Generate Plotly chart
        fig = px.bar(
            x=['Donations', 'Sustainability Updates'],
            y=[tx_counts.get('donation', 0), tx_counts.get('sustainability', 0)],
            labels={'x': 'Transaction Type', 'y': 'Count'},
            title='Blockchain Transactions by Type'
        )
        fig.update_layout(
            paper_bgcolor='#1A1A2E',
            plot_bgcolor='#1A1A2E',
            font_color='#00FFFF',
            title_font_color='#00FFFF'
        )
        fig.write_to_file("analytics/blockchain_transactions_by_store.html")
        
        analytics = {
            'transaction_counts': tx_counts,
            'donation_amount': -tx_amounts['stock_lbs'].get('donation', 0),
            'sustainability_score_sum': tx_amounts['sustainability_score'].get('sustainability', 0),
            'co2_reduction': -tx_amounts['stock_lbs'].get('donation', 0) * tx_amounts['co2_emission_factor'].get('donation', 0.1)
        }
        session.close()
        socketio.emit('analytics_update', {'store_id': store_id})
        return jsonify(analytics)
    except Exception as e:
        logging.error(f"Error generating blockchain analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/reports/<report_type>", methods=["POST"])
def generate_report(report_type):
    try:
        store_id = request.json.get("store_id", "store_1")
        session = Session()
        df_store = pd.read_sql(f"SELECT * FROM inventory WHERE store_id = '{store_id}'", engine)
        df_blockchain = pd.read_sql(f"SELECT * FROM inventory WHERE store_id = '{store_id}' AND tx_hash IS NOT NULL", engine)
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title = "Sustainability Report" if report_type == "sustainability" else "Blockchain Transaction Report"
        story.append(Paragraph(title, styles['Heading1']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Store ID: {store_id}", styles['Normal']))
        story.append(Spacer(1, 12))

        if report_type == "sustainability":
            story.append(Paragraph(f"Average Sustainability Score: {df_store['sustainability_score'].mean():.2f}", styles['Normal']))
            story.append(Paragraph(f"SDG Alignment: {df_store['sdg_alignment'].mean():.2f}", styles['Normal']))
            story.append(Paragraph(f"Waste Reduction Trend: {df_store['sustainability_score'].mean() * 100:.2f}%", styles['Normal']))
            story.append(Paragraph(f"CO2 Reduction: {(-df_store['stock_lbs'].where(df_store['transaction_type'] == 'donation').sum()) * df_store['co2_emission_factor'].mean():.2f} tons", styles['Normal']))
            story.append(Paragraph(f"Average Packaging Waste: {df_store['packaging_waste'].mean():.2f} lbs", styles['Normal']))
            story.append(Paragraph(f"Average Supply Chain Delay: {df_store['supply_chain_delay'].mean():.2f} days", styles['Normal']))
            story.append(Paragraph("Every donation reduces our carbon footprint, equivalent to planting trees!", styles['Normal']))
        elif report_type == "blockchain":
            tx_data = [[tx.item, abs(tx.stock_lbs) if tx.transaction_type == 'donation' else tx.sustainability_score, tx.transaction_type, tx.tx_hash, tx.date.strftime('%Y-%m-%d %H:%M:%S'), tx.co2_emission_factor] for tx in session.query(Inventory).filter(Inventory.store_id == store_id, Inventory.transaction_type.in_(["donation", "sustainability"])).all()]
            table = Table([['Item', 'Amount', 'Type', 'Tx Hash', 'Date', 'CO2 Factor']] + tx_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.cyan),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Total CO2 Reduction from Donations: {(-df_blockchain['stock_lbs'].where(df_blockchain['transaction_type'] == 'donation').sum()) * df_blockchain['co2_emission_factor'].mean():.2f} tons", styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        session.close()
        socketio.emit('analytics_update', {'store_id': store_id})
        return send_file(buffer, as_attachment=True, download_name=f"{report_type}_report_{store_id}.pdf", mimetype="application/pdf")
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        return jsonify({"error": str(e)}), 500

@socketio.on("update_inventory")
def handle_update(data):
    try:
        store_id = data.get("store_id", "store_1")
        session = Session()
        df = pd.read_sql(f"SELECT * FROM inventory WHERE store_id = '{store_id}'", engine)
        df["temperature_c"] += np.random.uniform(-2, 2)
        available_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
        X = df[available_columns]
        X_scaled = scaler.transform(X)
        n_samples = X_scaled.shape[0] - window_size + 1
        if n_samples > 0:
            X_windowed = np.array([X_scaled[i:i + window_size] for i in range(n_samples)])
            X_scaled_lstm = X_windowed.reshape(n_samples, window_size, -1)
        else:
            X_scaled_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

        lstm_pred = lstm_model.predict(X_scaled_lstm, verbose=0).flatten()
        gru_pred = gru_model(torch.FloatTensor(X_scaled_lstm)).detach().numpy().flatten()
        transformer_pred = transformer_model(torch.FloatTensor(X_scaled_lstm)).detach().numpy().flatten()
        xgb_pred = xgb_model.predict(X_scaled[max(0, window_size-1):])
        rf_pred = rf_model.predict(X_scaled[max(0, window_size-1):])
        final_pred = (lstm_pred + gru_pred + transformer_pred + xgb_pred + rf_pred) / 5

        df["predicted_spoilage_risk"] = np.concatenate([final_pred, np.full(len(df) - len(final_pred), np.nan)])
        df["recommended_stock_lbs"] = df["sales_lbs_daily"] * (1 - df["predicted_spoilage_risk"]) * 1.2
        df["action"] = df.apply(lambda row: "Discount" if row["expiry_date"] <= datetime.datetime.now() + datetime.timedelta(days=2) and row["predicted_spoilage_risk"] > 0.5 else "Donate" if row["expiry_date"] <= datetime.datetime.now() + datetime.timedelta(days=1) else "Keep", axis=1)

        for index, row in df.iterrows():
            session.query(Inventory).filter_by(id=row["id"]).update({
                "temperature_c": row["temperature_c"],
                "predicted_spoilage_risk": row["predicted_spoilage_risk"],
                "recommended_stock_lbs": row["recommended_stock_lbs"],
                "action": row["action"]
            })
        session.commit()
        socketio.emit("inventory_update", df.to_dict(orient="records"))
        socketio.emit('analytics_update', {'store_id': store_id})
        session.close()
    except Exception as e:
        logging.error(f"Error updating inventory: {str(e)}")
        socketio.emit("inventory_update", {"error": str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)