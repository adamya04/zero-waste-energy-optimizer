# backend/agentic_ai.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import pipeline
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Inventory
import pandas as pd
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
app = Flask(__name__)
# enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not set in .env")

try:
    chatbot = pipeline("text-generation", model="distilgpt2", token=hf_token)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", token=hf_token)
except Exception as e:
    logging.error(f"Error loading pipelines: {str(e)}")
    raise Exception(f"Error loading pipelines: {str(e)}")

engine = create_engine(os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tronwaste"))
Session = sessionmaker(bind=engine)

def query_agent(query):
    try:
        labels = ["inventory_status", "spoilage_analysis", "optimization_suggestion", "sustainability_metrics", "donation_potential", "sdg_alignment", "blockchain_info", "blockchain_analytics"]
        classification = classifier(query, candidate_labels=labels)
        top_label = classification["labels"][0]

        sustainability_quotes = [
            "Reducing waste today ensures a greener tomorrow.",
            "Every donation counts towards a sustainable future.",
            "AI and blockchain: Powering zero-waste solutions."
        ]

        session = Session()
        response = ""
        if top_label == "inventory_status":
            df = pd.read_sql("SELECT store_id, item, stock_lbs, co2_emission_factor, packaging_waste FROM inventory", engine)
            response = f"Inventory Status:\n{df[['store_id', 'item', 'stock_lbs', 'co2_emission_factor', 'packaging_waste']].to_dict(orient='records')[:5]}\n{sustainability_quotes[0]}"
        elif top_label == "spoilage_analysis":
            df = pd.read_sql("SELECT store_id, item, spoilage_rate, temperature_c, humidity_percent, pressure_mb, wind_speed_mps FROM inventory", engine)
            response = f"Spoilage Analysis:\n{df.groupby('store_id').agg({'spoilage_rate': 'mean', 'temperature_c': 'mean', 'humidity_percent': 'mean', 'pressure_mb': 'mean', 'wind_speed_mps': 'mean'}).to_dict()}\n{sustainability_quotes[1]}"
        elif top_label == "optimization_suggestion":
            df = pd.read_sql("SELECT store_id, item, expiry_date, predicted_spoilage_risk, action FROM inventory WHERE expiry_date <= CURRENT_DATE + INTERVAL '2 days'", engine)
            suggestions = df[df["predicted_spoilage_risk"] > 0.5]["store_id"].unique()
            response = f"Optimization: Consider {df['action'].value_counts().to_dict()} for stores {suggestions.tolist()}\n{sustainability_quotes[2]}"
        elif top_label == "sustainability_metrics":
            df = pd.read_sql("SELECT store_id, sustainability_score, sdg_alignment, co2_emission_factor, packaging_waste, supply_chain_delay FROM inventory", engine)
            response = f"Sustainability Metrics:\n{df.groupby('store_id').agg({'sustainability_score': 'mean', 'sdg_alignment': 'mean', 'co2_emission_factor': 'mean', 'packaging_waste': 'mean', 'supply_chain_delay': 'mean'}).to_dict()}\n{sustainability_quotes[0]}"
        elif top_label == "donation_potential":
            df = pd.read_sql("SELECT store_id, stock_lbs, co2_emission_factor FROM inventory WHERE expiry_date <= CURRENT_DATE + INTERVAL '1 day'", engine)
            total = df["stock_lbs"].sum()
            co2 = total * df["co2_emission_factor"].mean() if not df.empty else 0
            response = f"Donation Potential: {total:.2f} lbs, CO2 Reduction: {co2:.2f} tons\n{sustainability_quotes[1]}"
        elif top_label == "sdg_alignment":
            df = pd.read_sql("SELECT store_id, sdg_alignment FROM inventory", engine)
            response = f"SDG Alignment:\n{df.groupby('store_id')['sdg_alignment'].mean().to_dict()}\n{sustainability_quotes[2]}"
        elif top_label == "blockchain_info":
            df = pd.read_sql("SELECT store_id, tx_hash, transaction_type, item, stock_lbs, sustainability_score, co2_emission_factor FROM inventory WHERE tx_hash IS NOT NULL", engine)
            response = f"Blockchain Transactions (Recent 5):\n{df[['store_id', 'item', 'stock_lbs', 'sustainability_score', 'co2_emission_factor', 'tx_hash', 'transaction_type']].to_dict(orient='records')[:5]}\n{sustainability_quotes[0]}"
        elif top_label == "blockchain_analytics":
            df = pd.read_sql("SELECT store_id, transaction_type, stock_lbs, sustainability_score, co2_emission_factor FROM inventory WHERE tx_hash IS NOT NULL", engine)
            tx_counts = df.groupby('transaction_type').size().to_dict()
            co2 = -df['stock_lbs'].where(df['transaction_type'] == 'donation').sum() * df['co2_emission_factor'].mean() if not df.empty else 0
            response = f"Blockchain Analytics:\nDonations: {tx_counts.get('donation', 0)} transactions\nSustainability Updates: {tx_counts.get('sustainability', 0)} transactions\nCO2 Reduction: {co2:.2f} tons\n{sustainability_quotes[2]}"
        else:
            generated = chatbot(query, max_length=50, num_return_sequences=1)[0]["generated_text"]
            response = f"{generated.strip()}\n{sustainability_quotes[0]}"

        session.close()
        return response
    except Exception as e:
        logging.error(f"Error in query_agent: {str(e)}")
        return f"Error processing query: {str(e)}"

@app.route("/api/agent", methods=["POST"])
def agent_query():
    try:
        query = request.json.get("query", "")
        if not query:
            return jsonify({"error": "No query provided"}), 400
        response = query_agent(query)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error in agent_query: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
