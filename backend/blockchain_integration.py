# backend/blockchain_integration.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Inventory
import datetime
from stellar_sdk import Server, Keypair, TransactionBuilder, Network, Asset
import requests
import os
from dotenv import load_dotenv
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Stellar Testnet setup
HORIZON_URL = "https://horizon-testnet.stellar.org"
NETWORK_PASSPHRASE = Network.TESTNET_NETWORK_PASSPHRASE
try:
    OPERATOR_SECRET = os.getenv("STELLAR_OPERATOR_SECRET")
    OPERATOR_KEYPAIR = Keypair.from_secret(OPERATOR_SECRET)
    OPERATOR_PUBLIC = OPERATOR_KEYPAIR.public_key
except Exception as e:
    logging.error(f"Invalid STELLAR_OPERATOR_SECRET: {str(e)}")
    raise Exception(f"Invalid STELLAR_OPERATOR_SECRET: {str(e)}")

# Initialize Stellar server
server = Server(HORIZON_URL)

# Verify account is funded
def verify_account():
    try:
        logging.info(f"Verifying Stellar account: {OPERATOR_PUBLIC}")
        account = server.accounts().account_id(OPERATOR_PUBLIC).call()
        logging.info(f"Account {OPERATOR_PUBLIC} balance: {account['balances'][0]['balance']} XLM")
        return True
    except Exception as e:
        logging.error(f"Failed to verify account: {str(e)}")
        return False

# Fund account if needed
def fund_account():
    if not verify_account():
        logging.info(f"Funding account {OPERATOR_PUBLIC} on Testnet")
        try:
            response = requests.get(f"https://friendbot.stellar.org?addr={OPERATOR_PUBLIC}")
            response.raise_for_status()
            logging.info(f"Account funded: {response.json()}")
        except Exception as e:
            logging.error(f"Failed to fund account: {str(e)}")
            raise Exception(f"Failed to fund account: {str(e)}")

# Database setup
engine = create_engine(os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tronwaste"))
Session = sessionmaker(bind=engine)
session = Session()

def log_transaction(store_id, item, amount_lbs, transaction_type):
    try:
        logging.info(f"Logging transaction: {store_id}, {item}, {amount_lbs}, {transaction_type}")
        # Load account
        account = server.load_account(OPERATOR_PUBLIC)
        # Build transaction with memo
        memo = f"{store_id}:{item}:{amount_lbs}:{transaction_type}"
        transaction = (
            TransactionBuilder(
                source_account=account,
                network_passphrase=NETWORK_PASSPHRASE,
                base_fee=100
            )
            .append_payment_op(
                destination=OPERATOR_PUBLIC,
                asset=Asset.native(),
                amount="0.0000001"
            )
            .add_text_memo(memo[:28])
            .set_timeout(30)
            .build()
        )
        # Sign and submit
        transaction.sign(OPERATOR_KEYPAIR)
        response = server.submit_transaction(transaction)
        tx_hash = response["hash"]
        logging.info(f"Transaction executed: {tx_hash}")

        # Log to database
        new_entry = Inventory(
            store_id=store_id,
            item=item,
            date=datetime.datetime.now(),
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_mb=1013.0,
            wind_speed_mps=5.0,
            stock_lbs=-float(amount_lbs) if transaction_type == "donation" else 0,
            sales_lbs_daily=0,
            expiry_date=datetime.datetime.now() + datetime.timedelta(days=7),
            weather="Sunny",
            spoilage_rate=0.1,
            co2_emission_factor=0.5,
            supply_chain_delay=1.0,
            packaging_waste=0.0,
            transport_distance_km=100.0,
            sustainability_score=float(amount_lbs) if transaction_type == "sustainability" else 0,
            sdg_alignment=0.8,
            predicted_spoilage_risk=0.1,
            recommended_stock_lbs=0,
            action="Donate" if transaction_type == "donation" else "Keep",
            tx_hash=tx_hash,
            transaction_type=transaction_type
        )
        session.add(new_entry)
        session.commit()
        logging.info("Transaction logged in database")
        return tx_hash
    except Exception as e:
        session.rollback()
        logging.error(f"Transaction logging failed: {str(e)}")
        raise Exception(f"Transaction logging failed: {str(e)}")

@app.route("/api/blockchain/donate", methods=["POST"])
def donate():
    data = request.json
    store_id = data.get("store_id")
    item = data.get("item")
    amount_lbs = data.get("amount_lbs")
    if not all([store_id, item, amount_lbs]):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        amount_lbs = float(amount_lbs)
        if amount_lbs <= 0:
            return jsonify({"error": "Amount must be positive"}), 400
        tx_hash = log_transaction(store_id, item, amount_lbs, "donation")
        return jsonify({"tx_hash": tx_hash, "message": "Donation logged on Stellar Testnet"})
    except ValueError:
        return jsonify({"error": "Invalid amount_lbs format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/blockchain/sustainability", methods=["POST"])
def log_sustainability():
    data = request.json
    store_id = data.get("store_id")
    sustainability_score = data.get("sustainability_score")
    if not all([store_id, sustainability_score]):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        sustainability_score = float(sustainability_score)
        if sustainability_score < 0 or sustainability_score > 1:
            return jsonify({"error": "Sustainability score must be between 0 and 1"}), 400
        tx_hash = log_transaction(store_id, "sustainability_update", sustainability_score, "sustainability")
        return jsonify({"tx_hash": tx_hash, "message": "Sustainability update logged on Stellar Testnet"})
    except ValueError:
        return jsonify({"error": "Invalid sustainability_score format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/blockchain/transactions/<store_id>", methods=["GET"])
def get_transactions(store_id):
    try:
        transactions = session.query(Inventory).filter(
            Inventory.store_id == store_id,
            Inventory.transaction_type.in_(["donation", "sustainability"])
        ).all()
        tx_data = [
            {
                "store_id": t.store_id,
                "item": t.item,
                "amount_lbs": -t.stock_lbs if t.transaction_type == "donation" else t.sustainability_score,
                "timestamp": t.date.isoformat(),
                "tx_hash": t.tx_hash,
                "type": t.transaction_type,
                "co2_emission_factor": t.co2_emission_factor
            } for t in transactions
        ]
        return jsonify(tx_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    fund_account()
    app.run(debug=True, host="0.0.0.0", port=5002)
