
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
import json
import time
from datetime import datetime
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie

# Set page config for wide layout and sustainability icon
st.set_page_config(
    page_title="Zero Waste Energy Optimizer",
    page_icon="frontend/public/assets/eco-icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "quote_index" not in st.session_state:
    st.session_state.quote_index = 0
if "last_update" not in st.session_state:
    st.session_state.last_update = 0

# Theme CSS
def get_theme_css(theme):
    if theme == "dark":
        return """
            body, .stApp {
                background: linear-gradient(45deg, #0A0A23, #1A1A2E);
                background-size: 400%;
                animation: gradient 15s ease infinite;
                color: #00FFFF !important;
                font-family: 'Orbitron', sans-serif;
            }
            .stButton>button {
                background-color: #1A1A2E;
                color: #00FFFF;
                border: 2px solid #00FF00;
                border-radius: 8px;
                padding: 8px 16px;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                background-color: #00FF00;
                color: #0A0A23;
                box-shadow: 0 0 10px #00FF00;
            }
            .stTextInput>div>input, .stNumberInput>div>input {
                background-color: #1A1A2E;
                color: #00FFFF;
                border: 1px solid #00FF00;
                border-radius: 5px;
            }
            .stMetricLabel, .stMetricValue {
                color: #FF00FF !important;
                text-shadow: 0 0 5px #FF00FF;
            }
            .stDataFrame table {
                background-color: #1A1A2E !important;
                color: #00FFFF !important;
                border: 1px solid #00FF00;
            }
            .stPlotlyChart {
                background-color: #1A1A2E !important;
                border: 2px solid #00FF00;
                border-radius: 8px;
            }
        """
    else:
        return """
            body, .stApp {
                background: linear-gradient(45deg, #FFFFFF, #E0F7FA);
                background-size: 400%;
                animation: gradient 15s ease infinite;
                color: #008000 !important;
                font-family: 'Orbitron', sans-serif;
            }
            .stButton>button {
                background-color: #E0F7FA;
                color: #008000;
                border: 2px solid #00FF00;
                border-radius: 8px;
                padding: 8px 16px;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                background-color: #00FF00;
                color: #FFFFFF;
                box-shadow: 0 0 10px #00FF00;
            }
            .stTextInput>div>input, .stNumberInput>div>input {
                background-color: #E0F7FA;
                color: #008000;
                border: 1px solid #00FF00;
                border-radius: 5px;
            }
            .stMetricLabel, .stMetricValue {
                color: #008000 !important;
                text-shadow: 0 0 5px #00FF00;
            }
            .stDataFrame table {
                background-color: #E0F7FA !important;
                color: #008000 !important;
                border: 1px solid #00FF00;
            }
            .stPlotlyChart {
                background-color: #E0F7FA !important;
                border: 2px solid #00FF00;
                border-radius: 8px;
            }
        """

# Apply theme
heading_color = '#00FFFF' if st.session_state.theme == 'dark' else '#008000'
heading_shadow = '#00FFFF' if st.session_state.theme == 'dark' else '#00FF00'
sidebar_bg = '#1A1A2E' if st.session_state.theme == 'dark' else '#E0F7FA'
quote_color = '#FF00FF' if st.session_state.theme == 'dark' else '#008000'
tooltip_bg = '#1A1A2E' if st.session_state.theme == 'dark' else '#E0F7FA'
tooltip_color = '#00FFFF' if st.session_state.theme == 'dark' else '#008000'
expander_bg = '#1A1A2E' if st.session_state.theme == 'dark' else '#E0F7FA'
sidebar_text_color = '#00FFFF' if st.session_state.theme == 'dark' else '#008000'

st.markdown(f"""
    <style>
    {get_theme_css(st.session_state.theme)}
    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {heading_color} !important;
        text-shadow: 0 0 8px {heading_shadow};
    }}
    .stSidebar .sidebar-content {{
        background-color: {sidebar_bg};
        border-right: 2px solid #00FF00;
        border-radius: 8px;
        padding: 10px;
    }}
    .quote {{
        color: {quote_color};
        font-style: italic;
        text-align: center;
        margin: 30px 0;
        animation: fadeIn 1s ease-in-out;
    }}
    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: pointer;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 300px;
        background-color: {tooltip_bg};
        color: {tooltip_color};
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        border: 1px solid #00FF00;
        box-shadow: 0 0 5px #00FF00;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
    }}
    .stExpander {{
        background-color: {expander_bg};
        border: 1px solid #00FF00;
        border-radius: 8px;
        margin-bottom: 10px;
    }}
    .header-logo {{
        transition: transform 0.3s;
    }}
    .header-logo:hover {{
        transform: scale(1.1);
    }}
    .sidebar-text {{
        color: {sidebar_text_color};
        font-size: 14px;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 5px #00FF00;
    }}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Load Lottie animations
def load_lottie_file(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading Lottie animation: {str(e)}")
        return None

lottie_animation = load_lottie_file("D:/zero-waste-energy-optimizer/frontend/public/assets/eco-animation.json")
sdg_animation = load_lottie_file("D:/zero-waste-energy-optimizer/frontend/public/assets/sdg-animation.json")
carbon_animation = load_lottie_file("D:/zero-waste-energy-optimizer/frontend/public/assets/carbon-footprint.json")

# Rotating quotes
quotes = [
    "Reducing waste today ensures a greener tomorrow. - Sustainable Future",
    "Blockchain ensures transparency in every donation, powering trust. - Stellar Network",
    "AI predicts spoilage to save resources, ethically guiding sustainability. - Zero Waste",
    "Every action counts towards achieving SDG 12: Responsible Consumption. - United Nations",
    "Ethical AI drives decisions for a planet-first future. - Red Cross Mission"
]

def update_quote():
    st.session_state.quote_index = (st.session_state.quote_index + 1) % len(quotes)

# Backend API URLs
BASE_URL = "http://localhost:5000"
AGENT_URL = "http://localhost:5001"
BLOCKCHAIN_URL = "http://localhost:5002"

# Fetch functions
def fetch_inventory(store_id="store_1"):
    try:
        response = requests.get(f"{BASE_URL}/api/inventory/{store_id}", timeout=5)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Error fetching inventory: {str(e)}")
        return pd.DataFrame()

def fetch_analytics(store_id="store_1"):
    try:
        response = requests.get(f"{BASE_URL}/api/analytics/{store_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching analytics: {str(e)}")
        return {}

def fetch_blockchain_transactions(store_id="store_1"):
    try:
        response = requests.get(f"{BLOCKCHAIN_URL}/api/blockchain/transactions/{store_id}", timeout=5)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Error fetching blockchain transactions: {str(e)}")
        return pd.DataFrame()

def fetch_blockchain_analytics(store_id="store_1"):
    try:
        response = requests.get(f"{BASE_URL}/api/blockchain/analytics/{store_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching blockchain analytics: {str(e)}")
        return {}

def fetch_agent_response(query):
    try:
        response = requests.post(f"{AGENT_URL}/api/agent", json={"query": query}, timeout=5)
        response.raise_for_status()
        return response.json().get("response", "No response from AI agent.")
    except Exception as e:
        st.error(f"Error fetching AI agent response: {str(e)}")
        return "Error communicating with AI agent."

# Sidebar
with st.sidebar:
    st.image("D:/zero-waste-energy-optimizer/frontend/public/assets/store-icon.svg", width=40)
    st.markdown("<h3 style='margin: 10px 0;'>Navigate</h3>", unsafe_allow_html=True)
    page = st.selectbox(
        "Select Page",
        ["Dashboard", "Customer App", "Analytics", "AI Agent Chat", "Sustainability Tracker", "Report Generator", "Blockchain Analytics"],
        key="nav",
        label_visibility="collapsed"
    )
    st.markdown("<h3 style='margin: 10px 0;'>Theme</h3>", unsafe_allow_html=True)
    theme = st.selectbox(
        "Select Theme",
        ["Dark", "Bright"],
        key="theme_select",
        label_visibility="collapsed"
    )
    st.session_state.theme = theme.lower()
    st.markdown("---")
    st.markdown("<div class='sidebar-text'><strong>Zero Waste Energy Optimizer</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-text'>Join the sustainability revolution with AI and blockchain.</div>", unsafe_allow_html=True)
    if sdg_animation:
        st_lottie(sdg_animation, height=80, width=80, key="sidebar_sdg_lottie")

# Header
st.markdown("""
    <div style="text-align: center; padding: 20px 0; margin-bottom: 20px;">
        <img src="/assets/logo.png" alt="Zero Waste Energy Optimizer Logo" style="width: 60px; height: 60px;" class="header-logo">
        <h1 style='font-size: 28px; margin: 10px 0;'>Zero Waste Energy Optimizer</h1>
        <p style="color: #00FF00; font-size: 16px; text-shadow: 0 0 5px #00FF00; font-family: 'Orbitron', sans-serif;">
            AI & Blockchain for a Sustainable Future
        </p>
    </div>
""", unsafe_allow_html=True)
if lottie_animation:
    st_lottie(lottie_animation, height=80, width=80, key="header_lottie")
st.markdown(f'<p class="quote">{quotes[st.session_state.quote_index]}</p>', unsafe_allow_html=True)
if st.button("Next Quote", key="quote_button"):
    update_quote()

# Real-time updates (polling)
if time.time() - st.session_state.last_update > 5:
    st.session_state.last_update = time.time()
    st.rerun()

# Visualization descriptions
vis_descriptions = {
    "blockchain_transactions_by_store": "Bar chart showing the count of blockchain transactions by type (donations vs. sustainability updates).",
    "3d_spoilage_co2_sustainability": "3D scatter plot visualizing spoilage risk, CO2 emissions, and sustainability scores across stores.",
    "carbon_emission_trajectory": "Line chart tracking CO2 emission factors over time, highlighting reduction trends.",
    "carbon_footprint_spoilage": "Scatter plot comparing carbon footprint against spoilage rates for inventory items.",
    "correlation_heatmap": "Heatmap showing correlations between environmental factors (e.g., temperature, humidity) and spoilage.",
    "feature_distribution": "Distribution plots of key inventory metrics (e.g., stock levels, spoilage rates).",
    "global_sustainability": "Global map visualizing sustainability scores across different store locations.",
    "neon_pulse_waste_reduction": "Animated bar chart highlighting waste reduction achievements by item.",
    "sustainability_by_quarter": "Bar chart showing sustainability scores aggregated by quarter.",
    "sustainability_efficiency_heatmap": "Heatmap of sustainability efficiency across stores and time periods.",
    "sustainability_trajectory": "Line chart tracking sustainability score trends over time.",
    "supply_chain_delay_network": "Network graph illustrating supply chain delays and their impact on inventory.",
    "top_supply_efficiency_stores": "Bar chart ranking stores by supply chain efficiency.",
    "top_sustainability_stores": "Bar chart ranking stores by sustainability scores.",
    "waste_reduction_bars": "Bar chart showing waste reduction percentages by store or item.",
    "weather_impact_globe": "Interactive globe showing weather impacts on inventory sustainability.",
    "performance_metrics": "Table of performance metrics including stock levels, sales, and spoilage rates.",
    "supply_chain_optimization_report": "Detailed report on supply chain optimization metrics and recommendations.",
    "sustainability_metrics_report": "Comprehensive Excel report on sustainability metrics, including SDG alignment and CO2 reduction."
}

# Pages
if page == "Dashboard":
    st.header("Manager Dashboard")
    df = fetch_inventory()
    metrics = fetch_analytics()

    if not df.empty:
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.metric("Avg Sustainability Score", f"{metrics.get('sustainability_score_avg', 0):.2f}")
        with col2:
            st.metric("Donation Potential (lbs)", f"{metrics.get('donation_potential', 0):.2f}")
        with col3:
            st.metric("CO2 Reduction (tons)", f"{metrics.get('co2_reduction', 0):.2f}")

        st.subheader("Inventory Overview")
        with st.expander("View Inventory", expanded=True):
            st.dataframe(
                df[["item", "stock_lbs", "expiry_date", "predicted_spoilage_risk", "co2_emission_factor", "packaging_waste", "action"]]
                .assign(expiry_date=lambda x: pd.to_datetime(x["expiry_date"]).dt.strftime("%Y-%m-%d"))
                .style.set_properties(**{'background-color': '#1A1A2E' if st.session_state.theme == 'dark' else '#E0F7FA', 'color': '#00FFFF' if st.session_state.theme == 'dark' else '#008000', 'border-color': '#00FF00'})
            )
    else:
        st.warning("No inventory data available.")
    if lottie_animation:
        st_lottie(lottie_animation, height=80, width=80, key="dashboard_lottie")

elif page == "Customer App":
    st.header("Customer App")
    df = fetch_inventory()

    if not df.empty:
        st.subheader("Available Items")
        with st.expander("View Items", expanded=True):
            for _, row in df.iterrows():
                st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px; padding: 10px; border: 1px solid #00FF00; border-radius: 8px;">
                        <img src="/assets/{row['item'].lower()}.png" alt="{row['item']}" style="width: 40px; height: 40px; margin-right: 10px;" onerror="this.src='/assets/eco-icon.svg';">
                        <span>{row['item']} — {row['stock_lbs']:.2f} lbs — {row['action']}</span>
                    </div>
                """, unsafe_allow_html=True)

        st.subheader("Donate")
        with st.expander("Make a Donation"):
            item = st.text_input("Item", key="donation_item")
            amount_lbs = st.number_input("Amount (lbs)", min_value=0.0, step=0.01, key="donation_amount")
            if st.button("Donate"):
                if item and amount_lbs > 0:
                    try:
                        response = requests.post(f"{BLOCKCHAIN_URL}/api/blockchain/donate", json={
                            "store_id": "store_1",
                            "item": item,
                            "amount_lbs": amount_lbs
                        }, timeout=5)
                        response.raise_for_status()
                        st.success(f"Donation successful! Tx Hash: {response.json().get('tx_hash')}")
                    except Exception as e:
                        st.error(f"Donation failed: {str(e)}")
                else:
                    st.error("Please enter a valid item and amount.")
    else:
        st.warning("No inventory data available.")
    if carbon_animation:
        st_lottie(carbon_animation, height=80, width=80, key="customer_lottie")

elif page == "Analytics":
    st.header("Analytics Dashboard")
    visualizations = [
        {"key": "blockchain_transactions_by_store", "type": "html", "url": f"{BASE_URL}/analytics/blockchain_transactions_by_store.html", "title": "Blockchain Transactions by Type"},
        {"key": "3d_spoilage_co2_sustainability", "type": "image", "url": f"{BASE_URL}/analytics/3d_spoilage_co2_sustainability.png", "title": "3D Spoilage & CO2 Sustainability"},
        {"key": "carbon_emission_trajectory", "type": "html", "url": f"{BASE_URL}/analytics/carbon_emission_trajectory.html", "title": "Carbon Emission Trajectory"},
        {"key": "carbon_footprint_spoilage", "type": "image", "url": f"{BASE_URL}/analytics/carbon_footprint_spoilage.png", "title": "Carbon Footprint vs. Spoilage"},
        {"key": "correlation_heatmap", "type": "image", "url": f"{BASE_URL}/analytics/correlation_heatmap.png", "title": "Correlation Heatmap"},
        {"key": "feature_distribution", "type": "image", "url": f"{BASE_URL}/analytics/feature_distribution.png", "title": "Feature Distribution"},
        {"key": "global_sustainability", "type": "html", "url": f"{BASE_URL}/analytics/global_sustainability.html", "title": "Global Sustainability"},
        {"key": "neon_pulse_waste_reduction", "type": "image", "url": f"{BASE_URL}/analytics/neon_pulse_waste_reduction.png", "title": "Neon Pulse Waste Reduction"},
        {"key": "sustainability_by_quarter", "type": "image", "url": f"{BASE_URL}/analytics/sustainability_by_quarter.png", "title": "Sustainability by Quarter"},
        {"key": "sustainability_efficiency_heatmap", "type": "image", "url": f"{BASE_URL}/analytics/sustainability_efficiency_heatmap.png", "title": "Sustainability Efficiency Heatmap"},
        {"key": "sustainability_trajectory", "type": "html", "url": f"{BASE_URL}/analytics/sustainability_trajectory.html", "title": "Sustainability Trajectory"},
        {"key": "supply_chain_delay_network", "type": "html", "url": f"{BASE_URL}/analytics/supply_chain_delay_network.html", "title": "Supply Chain Delay Network"},
        {"key": "top_supply_efficiency_stores", "type": "image", "url": f"{BASE_URL}/analytics/top_supply_efficiency_stores.png", "title": "Top Supply Efficiency Stores"},
        {"key": "top_sustainability_stores", "type": "image", "url": f"{BASE_URL}/analytics/top_sustainability_stores.png", "title": "Top Sustainability Stores"},
        {"key": "waste_reduction_bars", "type": "html", "url": f"{BASE_URL}/analytics/waste_reduction_bars.html", "title": "Waste Reduction Bars"},
        {"key": "weather_impact_globe", "type": "html", "url": f"{BASE_URL}/analytics/weather_impact_globe.html", "title": "Weather Impact Globe"},
        {"key": "performance_metrics", "type": "csv", "url": f"{BASE_URL}/analytics/performance_metrics.csv", "title": "Performance Metrics"},
        {"key": "supply_chain_optimization_report", "type": "csv", "url": f"{BASE_URL}/analytics/supply_chain_optimization_report.csv", "title": "Supply Chain Optimization Report"},
        {"key": "sustainability_metrics_report", "type": "excel", "url": f"{BASE_URL}/analytics/sustainability_metrics_report.xlsx", "title": "Sustainability Metrics Report"}
    ]

    for vis in visualizations:
        with st.expander(vis["title"], expanded=False):
            st.markdown(f"""
                <div class="tooltip">
                    <h4 style="margin: 5px 0;">{vis["title"]}</h4>
                    <span class="tooltiptext">{vis_descriptions[vis["key"]]}</span>
                </div>
            """, unsafe_allow_html=True)
            try:
                if vis["type"] == "html":
                    components.iframe(vis["url"], height=600, scrolling=True)
                elif vis["type"] == "image":
                    st.image(vis["url"], caption=vis["title"], use_column_width=True)
                elif vis["type"] == "csv":
                    df = pd.read_csv(vis["url"])
                    st.dataframe(df.style.set_properties(**{'background-color': '#1A1A2E' if st.session_state.theme == 'dark' else '#E0F7FA', 'color': '#00FFFF' if st.session_state.theme == 'dark' else '#008000', 'border-color': '#00FF00'}))
                    if len(df.columns) >= 2:
                        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=vis["title"], template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white")
                        fig.update_layout(
                            plot_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
                            paper_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
                            font_color="#00FFFF" if st.session_state.theme == "dark" else "#008000",
                            title_font_color="#00FF00"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                elif vis["type"] == "excel":
                    df = pd.read_excel(vis["url"])
                    st.dataframe(df.style.set_properties(**{'background-color': '#1A1A2E' if st.session_state.theme == 'dark' else '#E0F7FA', 'color': '#00FFFF' if st.session_state.theme == 'dark' else '#008000', 'border-color': '#00FF00'}))
                    if len(df.columns) >= 2:
                        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title=vis["title"], template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white")
                        fig.update_layout(
                            plot_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
                            paper_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
                            font_color="#00FFFF" if st.session_state.theme == "dark" else "#008000",
                            title_font_color="#00FF00"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading {vis['title']}: {str(e)}")
    if lottie_animation:
        st_lottie(lottie_animation, height=80, width=80, key="analytics_lottie")

elif page == "AI Agent Chat":
    st.header("AI Agent Chat")
    with st.expander("Ask the AI Agent", expanded=True):
        query = st.text_input("Ask about sustainability, inventory, or blockchain:", key="agent_query")
        if st.button("Submit Query"):
            if query:
                response = fetch_agent_response(query)
                st.markdown(f"**AI Response**: {response}")
            else:
                st.error("Please enter a query.")
    if sdg_animation:
        st_lottie(sdg_animation, height=80, width=80, key="chat_lottie")

elif page == "Sustainability Tracker":
    st.header("Sustainability Tracker")
    metrics = fetch_analytics()

    if metrics:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.subheader("Key Performance")
            st.metric("Donation Potential (lbs)", f"{metrics.get('donation_potential', 0):.2f}")
            st.metric("Sustainability Score", f"{metrics.get('sustainability_score_avg', 0):.2f}")
            st.metric("SDG Alignment", f"{metrics.get('sdg_alignment_avg', 0):.2f}")
        with col2:
            st.subheader("SDG Goals")
            st.metric("Goal 12 (Responsible Consumption)", f"{metrics.get('sdg_alignment_avg', 0):.2f}")
            st.metric("Goal 13 (Climate Action)", f"{metrics.get('waste_reduction_trend', 0):.2f}%")

        st.subheader("SDG Alignment Wheel")
        sdg_data = pd.DataFrame({
            "SDG": ["Goal 12", "Goal 13", "Goal 15"],
            "Alignment": [metrics.get('sdg_alignment_avg', 0), metrics.get('waste_reduction_trend', 0) / 100, 0.7]
        })
        fig = px.pie(sdg_data, names="SDG", values="Alignment", title="SDG Alignment", template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white")
        fig.update_traces(marker=dict(colors=["#00FF00", "#00FFFF", "#FF00FF"]))
        fig.update_layout(
            plot_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
            paper_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
            font_color="#00FFFF" if st.session_state.theme == "dark" else "#008000",
            title_font_color="#00FF00"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("CO2 Reduction Counter")
        st.markdown(f"""
            <div style="text-align: center; font-size: 20px; color: #00FF00; text-shadow: 0 0 5px #00FF00;">
                {metrics.get('co2_reduction', 0):.2f} tons CO2 reduced
            </div>
        """, unsafe_allow_html=True)
        if carbon_animation:
            st_lottie(carbon_animation, height=80, width=80, key="co2_lottie")

        df = fetch_inventory()
        if not df.empty:
            with st.expander("Sustainability Score Trend"):
                fig = px.line(
                    x=df["date"],
                    y=df["sustainability_score"],
                    title="Sustainability Score Trend",
                    labels={"x": "Date", "y": "Sustainability Score"},
                    template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white"
                )
                fig.update_traces(line_color="#00FF00")
                fig.update_layout(
                    plot_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
                    paper_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
                    font_color="#00FFFF" if st.session_state.theme == "dark" else "#008000",
                    title_font_color="#00FF00"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No sustainability metrics available.")
    if sdg_animation:
        st_lottie(sdg_animation, height=80, width=80, key="sustainability_lottie")

elif page == "Report Generator":
    st.header("Report Generator")
    with st.expander("Generate a Report"):
        report_type = st.selectbox("Select Report Type", ["sustainability", "blockchain"], key="report_type")
        if st.button("Generate Report"):
            try:
                response = requests.post(f"{BASE_URL}/api/reports/{report_type}", json={"store_id": "store_1"}, stream=True)
                response.raise_for_status()
                st.download_button(
                    label="Download Report",
                    data=response.content,
                    file_name=f"{report_type}_report_store_1.pdf",
                    mime="application/pdf"
                )
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    if lottie_animation:
        st_lottie(lottie_animation, height=80, width=80, key="report_lottie")

elif page == "Blockchain Analytics":
    st.header("Blockchain Analytics")
    df = fetch_blockchain_transactions()
    metrics = fetch_blockchain_analytics()

    if not df.empty:
        with st.expander("Transaction Summary", expanded=True):
            st.dataframe(
                df[["item", "amount_lbs", "type", "timestamp", "tx_hash", "co2_emission_factor"]]
                .style.set_properties(**{'background-color': '#1A1A2E' if st.session_state.theme == 'dark' else '#E0F7FA', 'color': '#00FFFF' if st.session_state.theme == 'dark' else '#008000', 'border-color': '#00FF00'})
            )
        with st.expander("Blockchain Metrics"):
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.metric("Donation Amount (lbs)", f"{metrics.get('donation_amount', 0):.2f}")
            with col2:
                st.metric("CO2 Reduction (tons)", f"{metrics.get('co2_reduction', 0):.2f}")

            tx_counts = df.groupby("type").size().reset_index(name="count")
            fig = px.bar(
                tx_counts,
                x="type",
                y="count",
                title="Blockchain Transactions by Type",
                labels={"type": "Transaction Type", "count": "Count"},
                template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white"
            )
            fig.update_traces(marker_color="#00FF00")
            fig.update_layout(
                plot_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
                paper_bgcolor="#1A1A2E" if st.session_state.theme == "dark" else "#E0F7FA",
                font_color="#00FFFF" if st.session_state.theme == "dark" else "#008000",
                title_font_color="#00FF00"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No blockchain transactions available.")
    if carbon_animation:
        st_lottie(carbon_animation, height=80, width=80, key="blockchain_lottie")

# Footer
st.markdown(f"""
    <div style="text-align: center; color: {'#00FFFF' if st.session_state.theme == 'dark' else '#008000'}; margin: 30px 0; text-shadow: 0 0 5px #00FF00; font-size: 14px;">
        Created by Adamya Sharma | Powered by AI and ML | Join the Zero Waste Revolution
    </div>
""", unsafe_allow_html=True)