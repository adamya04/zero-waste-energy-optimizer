# Zero Waste Energy Optimizer

![Zero Waste Energy Optimizer Logo](frontend/public/assets/logo.png)

A Streamlit-based app leveraging **AI**, **blockchain**, and **multimodal analytics** to optimize inventory, reduce food waste, and align with UN SDGs 12 (Responsible Consumption) and 13 (Climate Action). It integrates predictive AI, Stellar Testnet blockchain, real-time weather data, and a chatbot for sustainable supply chain management.

## Features

- **AI-Powered Predictions**: LSTM, GRU, Transformer, XGBoost, and RandomForest models predict spoilage risks, cutting waste by up to 20% in tests.
- **AI Chatbot**: Hugging Face-powered (DistilGPT2, BART) chatbot provides real-time insights on inventory, spoilage, and sustainability.
- **Blockchain Transparency**: Tracks donations and sustainability updates on Stellar Testnet (`http://localhost:5002`).
- **Multimodal Analytics**: 19 Tron-inspired visualizations (3D plots, interactive globes, neon bar charts) using Plotly.
- **Sustainability Tracking**: Monitors CO2 reduction (e.g., 0.85 tons saved) and SDG alignment.
- **Reports**: Generates PDF reports for ESG compliance.
- **UI**: Neon-green theme (`#00FF00`), Orbitron font, compact logo (60x60px), 80x80px Lottie animations, rounded buttons, and enhanced sidebar with glowing text.

## Tech Stack

- **Frontend**: Streamlit, Plotly, streamlit-lottie
- **Backend**: Flask (APIs: `http://localhost:5000`, `5001`, `5002`), WebSockets
- **AI**: Hugging Face (DistilGPT2, BART), TensorFlow, PyTorch
- **Data**: Pandas, NumPy, OpenPyXL
- **APIs**: OpenWeather, Kaggle, Flask APIs
- **Blockchain**: Stellar Testnet, Web3.py
- **Database**: PostgreSQL
- **Assets**: Lottie animations (eco-animation.json, sdg-animation.json, carbon-footprint.json), SVG icons, PNGs
- **Styling**: Custom CSS, neon-green theme with gradient animations
- **Deployment**: Local (`http://localhost:8501`)

## Prerequisites

- Python 3.11+
- Git
- PostgreSQL
- OpenWeather API key
- Kaggle API key
- Hugging Face API token
- Stellar Testnet account
- Node.js (optional, for Lottie animations)

## Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/adamya04/zero-waste-energy-optimizer.git
   cd zero-waste-energy-optimizer


Set Up Virtual Environment:
python -m venv backend/venv
source backend/venv/bin/activate  # Linux/Mac
backend\venv\Scripts\activate  # Windows


Install Dependencies:
pip install -r backend/requirements.txt
pip install streamlit pandas numpy requests plotly streamlit-lottie openpyxl flask flask-socketio psycopg2-binary web3 transformers torch stellar-sdk


Configure Environment:

Create backend/.env (not tracked):OPENWEATHER_API_KEY=your_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
HUGGINGFACE_TOKEN=your_hf_token
STELLAR_OPERATOR_SECRET=your_stellar_secret
DATABASE_URL=postgresql://user:password@localhost:5432/zero_waste
FLASK_ENV=development




Set Up PostgreSQL:
psql -U postgres -c "CREATE DATABASE zero_waste;"
cd backend
python database_setup.py

Add lat and lon:
ALTER TABLE inventory ADD COLUMN lat FLOAT, ADD COLUMN lon FLOAT;
UPDATE inventory SET lat = 40.7128, lon = -74.0060 WHERE store_id = 'store_1';


Generate Synthetic Data:
cd backend
python data_collection.py



Running the Application

Start Backend Servers (three terminals):
cd backend
source venv/bin/activate
python app.py
python agentic_ai.py
python blockchain_integration.py

Verify:
curl http://localhost:5000/api/inventory/store_1
curl http://localhost:5001/api/agent -H "Content-Type: application/json" -d '{"query": "What is the spoilage risk?"}'
curl http://localhost:5002/api/blockchain/transactions/store_1


Start Streamlit:
cd frontend
source ../backend/venv/bin/activate
streamlit run streamlit_app.py

Access: http://localhost:8501


Usage

Dashboard: Real-time inventory, sustainability scores, CO2 reduction metrics.
Customer App: Browse items, make blockchain-tracked donations with animated feedback.
Analytics: 19 visualizations (3D scatter plots, heatmaps, interactive globes) with tooltips.
AI Chatbot: Query sustainability insights via Hugging Face-powered agent.
Sustainability Tracker: Monitors SDG alignment and CO2 reduction with pie charts and counters.
Reports: Download PDF sustainability/blockchain reports.
Blockchain Analytics: View transaction summaries and metrics with neon-styled bar charts.

Data Sources

Synthetic Data: Generated via data_collection.py.
OpenWeather API: Real-time weather data for spoilage predictions.
Kaggle API: Synthetic datasets for training AI models.
Blockchain: Stellar Testnet ledger for donation tracking.



License
MIT License. See LICENSE.

Contact: Adamya Sharma (adamyasharma7476@gmail.com)
GitHub: adamya04
Demo: http://localhost:8501```