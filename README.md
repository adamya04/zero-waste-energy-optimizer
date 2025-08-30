Zero Waste Energy Optimizer

A Streamlit-based application leveraging AI, blockchain, and multimodal analytics to optimize inventory management, reduce food waste, and align with UN Sustainable Development Goals (SDGs) 12 (Responsible Consumption) and 13 (Climate Action). This project integrates predictive modeling, real-time weather data, blockchain transparency, and agentic AI to drive sustainable supply chains.
Features

Predictive Modeling & Machine Learning: Uses LSTM, GRU, Transformer, XGBoost, and RandomForest models to predict spoilage risks, reducing waste by up to 20% in simulated tests.
OpenWeather Integration: Fetches real-time weather data (temperature, humidity) to assess environmental impacts on inventory.
Blockchain Transparency: Tracks donations and sustainability actions via a Web3-based ledger (http://localhost:5002), ensuring trust and auditability.
Agentic AI: Interactive chat interface (Hugging Face-based) for real-time insights on inventory and sustainability metrics.
Multimodal Analytics: Displays 19 visualizations (e.g., 3D spoilage-CO2 plots, global sustainability maps) using Plotly, served from backend/analytics/.
Sustainability Tracking: Monitors CO2 reduction (e.g., 0.85 tons saved) and SDG alignment via an interactive SDG wheel.
Report Generation: Produces PDF reports of blockchain transactions and sustainability metrics for ESG compliance.
Customizable UI: Features dark/bright themes, neon-green accents (#00FF00), Orbitron font, and compact Lottie animations (80x80px).
Real-Time Updates: Polls backend APIs every 5 seconds for live data.

Tech Stack

Frontend: Streamlit, Plotly, streamlit-lottie
Backend: Flask (APIs: http://localhost:5000, 5001, 5002), WebSockets
Data Processing: Pandas, NumPy, OpenPyXL
APIs: OpenWeather API, Kaggle API, custom Flask APIs
Blockchain: Web3.py for donation/sustainability tracking
Database: PostgreSQL (inventory, transactions)
Assets: Lottie animations, SVG icons, PNG visualizations
Styling: Custom CSS, Orbitron font, neon-green theme
Deployment: Local hosting (http://localhost:8501)

Prerequisites

Python 3.11+
Git
PostgreSQL
OpenWeather API key
Kaggle API key (for synthetic data)
Node.js (optional, for Lottie animations)

Setup

Clone the Repository:
git clone https://github.com/yourusername/zero-waste-energy-optimizer.git
cd zero-waste-energy-optimizer


Set Up Virtual Environment:
python -m venv backend/venv
source backend/venv/bin/activate  # Linux/Mac
backend\venv\Scripts\activate  # Windows


Install Dependencies:
pip install -r backend/requirements.txt

If requirements.txt is missing, install manually:
pip install streamlit pandas numpy requests plotly streamlit-lottie openpyxl flask flask-socketio psycopg2-binary web3


Configure Environment:

Create .env in backend/ (not tracked):OPENWEATHER_API_KEY=your_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
DATABASE_URL=postgresql://user:password@localhost:5432/zero_waste
FLASK_ENV=development


Get API keys from OpenWeather and Kaggle.


Set Up PostgreSQL:

Install PostgreSQL and create a database:psql -U postgres -c "CREATE DATABASE zero_waste;"


Run database_setup.py:cd backend
python database_setup.py


Add lat and lon columns to inventory table:ALTER TABLE inventory ADD COLUMN lat FLOAT, ADD COLUMN lon FLOAT;
UPDATE inventory SET lat = 40.7128, lon = -74.0060 WHERE store_id = 'store_1';




Verify Assets:

Ensure frontend/public/assets/ contains:
logo.png, eco-icon.svg, store-icon.svg, chat-icon.svg
banana.png, apple.png, berries.png, lettuce.png
eco-animation.json, sdg-animation.json, carbon-footprint.json (from LottieFiles)


Ensure backend/analytics/ contains:
PNGs: 3d_spoilage_co2_sustainability.png, etc.
HTMLs: carbon_emission_trajectory.html, etc.
CSVs/Excel: performance_metrics.csv, sustainability_metrics_report.xlsx




Generate Synthetic Data (if needed):
cd backend
python data_simulation.py



Running the Application

Start Backend Servers:Open three terminals:
cd backend
source venv/bin/activate  # or backend\venv\Scripts\activate
python app.py

cd backend
source venv/bin/activate
python agentic_ai.py

cd backend
source venv/bin/activate
python blockchain_integration.py

Verify endpoints:
curl http://localhost:5000/api/inventory/store_1
curl http://localhost:5001/api/agent -H "Content-Type: application/json" -d '{"query": "What is the spoilage risk?"}'
curl http://localhost:5002/api/blockchain/transactions/store_1


Start Streamlit:
cd frontend
source ../backend/venv/bin/activate  # or backend\venv\Scripts\activate
streamlit run streamlit_app.py

Access at http://localhost:8501. If port 8501 is busy:
streamlit run streamlit_app.py --server.port 8502



Usage

Dashboard: View real-time inventory, sustainability scores, donation potential, and CO2 reduction.
Customer App: Browse items and make blockchain-tracked donations.
Analytics: Explore 19 visualizations (e.g., 3D plots, heatmaps) with tooltips.
AI Agent Chat: Query sustainability or inventory insights (e.g., “What’s the CO2 reduction?”).
Sustainability Tracker: Monitor SDG alignment and CO2 savings via interactive charts.
Report Generator: Download PDF reports for sustainability and blockchain metrics.
Blockchain Analytics: View transaction logs and CO2 impact.

Data Sources

Synthetic Data: Simulated via data_simulation.py (replaceable with IoT sensor data for temperature/humidity).
OpenWeather API: Real-time weather data for environmental impact analysis.
Kaggle API: Synthetic datasets for model training.
Blockchain: Web3-based ledger for donation and sustainability tracking.

Folder Structure
zero-waste-energy-optimizer/
├── backend/
│   ├── analytics/              # PNG, HTML, CSV, Excel visualizations
│   ├── app.py                 # Main Flask API with WebSockets
│   ├── data_collection.py     # Kaggle API, IoT data
│   ├── iot_integration.py     # OpenWeatherMap API
│   ├── data_cleaning.py       # Data cleaning
│   ├── feature_engineering.py # Feature engineering
│   ├── feature_selection.py   # Feature selection
│   ├── model_development.py   # Multimodal AI models
│   ├── model_evaluation.py    # Evaluation with visualizations
│   ├── database_setup.py      # PostgreSQL setup
│   ├── data_simulation.py     # Synthetic data generation
│   ├── deploy_azure.py        # Azure deployment
│   ├── agentic_ai.py          # Hugging Face agent
│   ├── blockchain_integration.py # Web3 donation/sustainability tracking
│   └── requirements.txt       # Dependencies
├── frontend/
│   ├── public/
│   │   └── assets/            # Images and Lottie animations
│   │       ├── banana.png
│   │       ├── apple.png
│   │       ├── berries.png
│   │       ├── lettuce.png
│   │       ├── logo.png
│   │       ├── eco-icon.svg
│   │       ├── store-icon.svg
│   │       ├── chat-icon.svg
│   │       ├── eco-animation.json
│   │       ├── sdg-animation.json
│   │       ├── carbon-footprint.json
│   │   ├── particle-bg.js     # Optional JS for background effects
│   └── streamlit_app.py       # Streamlit frontend
├── README.md                  # Instructions
└── .gitignore                 # Ignore file

Contributing

Fork the repository.
Create a feature branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add your feature".
Push to branch: git push origin feature/your-feature.
Open a pull request with a detailed description.

Please follow the Code of Conduct and ensure tests pass:
pytest backend/tests/

License
MIT License. See LICENSE for details.
Acknowledgments
Inspired by sustainability advocates like Sasha Luciani (xAI) and Bill Weihl (ClimateVoice). Built with ❤️ for a zero-waste future.

Contact: [Your Name] (your.email@example.com)GitHub: yourusernameDemo: Access at http://localhost:8501 after setup.