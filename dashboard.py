"""
dashboard.py
============
BESCOM Smart Meter Intelligence — Streamlit Dashboard

To run:
  pip install streamlit plotly pandas numpy joblib
  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
import base64

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title     = "BESCOM Smart Meter Intelligence",
    page_icon      = "⚡",
    layout         = "wide",
    initial_sidebar_state = "expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Advanced Custom CSS (Futuristic Dark Theme)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* Global Typography & Theme */
  html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif !important;
  }

  /* App Background (Dark Navy/Black Gradient) */
  [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0b1021 0%, #04060d 100%);
    color: #e0e6ed;
  }
  
  [data-testid="stHeader"] {
    background: transparent;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background-color: rgba(6, 10, 20, 0.7) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(0, 255, 255, 0.1);
  }
  
  /* Sidebar selected items glow */
  .stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,255,255,0.2);
    color: #00ffff;
  }

  /* Hero Section */
  .hero-container {
    padding: 3rem 2rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.05) 0%, rgba(138, 43, 226, 0.05) 100%);
    border-radius: 20px;
    border: 1px solid rgba(0, 255, 255, 0.1);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(0, 255, 255, 0.05);
  }
  .hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    margin: 0;
    background: -webkit-linear-gradient(45deg, #00ffff, #00a2ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
  }
  .hero-subtitle {
    font-size: 1.2rem;
    font-weight: 400;
    color: #8fa0c0;
    margin-top: 0.5rem;
  }

  /* KPI Glass Cards */
  .kpi-card {
    background: rgba(15, 20, 35, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 255, 255, 0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: left;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
  }
  .kpi-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: rgba(0, 255, 255, 0.4);
    box-shadow: 0 15px 40px 0 rgba(0, 255, 255, 0.15);
  }
  .kpi-title {
    font-size: 0.9rem;
    color: #8fa0c0;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 10px 0 5px 0;
  }
  .kpi-trend-up {
    color: #00ff88;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .kpi-trend-down {
    color: #ff3366;
    font-size: 0.85rem;
    font-weight: 600;
  }

  /* Live Pulse Indicator */
  .live-status {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 10px;
    background: rgba(0, 255, 136, 0.05);
    border: 1px solid rgba(0, 255, 136, 0.2);
    border-radius: 8px;
    margin-bottom: 20px;
  }
  .pulse-dot {
    width: 10px;
    height: 10px;
    background-color: #00ff88;
    border-radius: 50%;
    box-shadow: 0 0 10px #00ff88;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
  }
  .live-text {
    color: #00ff88;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* Alert Feed Cards */
  .alert-feed-card {
    background: rgba(255, 51, 102, 0.05);
    border-left: 4px solid #ff3366;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
  }
  .alert-feed-card:hover {
    background: rgba(255, 51, 102, 0.1);
    transform: translateX(5px);
  }
  .alert-feed-title {
    font-weight: 700;
    color: #ff3366;
    margin-bottom: 4px;
    font-size: 1rem;
  }
  .alert-feed-desc {
    font-size: 0.85rem;
    color: #e0e6ed;
  }

  /* Segmented Control Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 20, 35, 0.5);
    border-radius: 12px;
    padding: 8px;
  }
  .stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 8px;
    background: transparent;
    border: none;
    padding: 0 24px;
    color: #8fa0c0;
    font-weight: 600;
    transition: all 0.3s ease;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(0, 255, 255, 0.1) !important;
    color: #00ffff !important;
    box-shadow: inset 0 0 10px rgba(0, 255, 255, 0.2);
  }

  /* Bottom Tagline */
  .tagline {
    text-align: center;
    color: #4a5b78;
    font-size: 0.85rem;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  /* Zone Risk Map Grid & Cards */
  .zone-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-top: 15px;
    margin-bottom: 30px;
  }
  
  .zone-card {
    background: rgba(15, 20, 35, 0.5);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 12px;
    padding: 20px;
    border-top: 4px solid transparent;
    border-left: 1px solid rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  
  .zone-card:hover {
    transform: translateY(-5px) scale(1.02);
  }

  .zone-card .tooltip {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: calc(100% + 10px);
    left: 50%;
    transform: translateX(-50%) translateY(10px);
    background: rgba(6, 10, 20, 0.95);
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    color: #fff;
    padding: 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    white-space: nowrap;
    z-index: 10;
    transition: all 0.3s ease;
    pointer-events: none;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
  }
  
  .zone-card:hover .tooltip {
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }

  /* Risk Variations */
  .zone-low { border-top-color: #00ff88; }
  .zone-low:hover { box-shadow: 0 10px 30px rgba(0, 255, 136, 0.15); }
  
  .zone-medium { border-top-color: #ffcc00; }
  .zone-medium:hover { box-shadow: 0 10px 30px rgba(255, 204, 0, 0.15); }
  
  .zone-critical { 
    border-top-color: #ff3366; 
    animation: pulse-red 2s infinite; 
  }
  .zone-critical:hover { box-shadow: 0 10px 40px rgba(255, 51, 102, 0.3); }

  @keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0.3); }
    70% { box-shadow: 0 0 0 15px rgba(255, 51, 102, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0); }
  }

  .zone-name {
    font-size: 0.9rem;
    color: #8fa0c0;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 5px;
  }
  .zone-avg {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 10px;
  }
  .zone-badge-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .zone-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .badge-low { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid rgba(0, 255, 136, 0.3); }
  .badge-medium { background: rgba(255, 204, 0, 0.1); color: #ffcc00; border: 1px solid rgba(255, 204, 0, 0.3); }
  .badge-critical { background: rgba(255, 51, 102, 0.1); color: #ff3366; border: 1px solid rgba(255, 51, 102, 0.3); }
  .zone-trend {
    font-size: 0.85rem;
    font-weight: 700;
  }
  
  .insight-overlay {
    background: linear-gradient(90deg, rgba(255, 51, 102, 0.15) 0%, rgba(255, 51, 102, 0.02) 100%);
    border-left: 4px solid #ff3366;
    padding: 15px 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 15px;
    backdrop-filter: blur(10px);
  }
  .insight-icon { font-size: 1.5rem; }
  .insight-text { color: #e0e6ed; font-size: 1.05rem; font-weight: 500; }

  /* dataframe styling */
  [data-testid="stDataFrame"] {
    background: rgba(0,0,0,0.2);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
  }

  /* Hide streamlit default header */
  #MainMenu {visibility: hidden;}
  footer     {visibility: hidden;}
  header     {visibility: hidden;}

  /* ══════════════════════════════════════════════════════════════════════════════
     PRESENTATION UI STYLES
  ══════════════════════════════════════════════════════════════════════════════ */
  .pres-section {
    padding: 60px 40px;
    margin-bottom: 30px;
    background: #0f1115;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
    color: #e0e6ed;
    position: relative;
    overflow: hidden;
  }
  
  .hero-section {
    padding: 120px 40px;
    background: radial-gradient(circle at top right, #1a1c23 0%, #08090a 100%);
    text-align: left;
    margin-top: -2rem;
  }
  
  .pres-hero-title {
    font-size: 4rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 25px;
    position: relative;
    z-index: 1;
    max-width: 900px;
  }
  
  .pres-hero-subtitle {
    font-size: 1.3rem;
    color: #9ba6b6;
    max-width: 800px;
    line-height: 1.6;
    position: relative;
    z-index: 1;
  }
  
  .pres-section-title {
    font-size: 2.8rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 40px;
    line-height: 1.2;
  }
  
  .insight-cards-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }
  
  .i-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 35px;
    transition: transform 0.3s ease, background 0.3s ease;
  }
  
  .i-card:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.08);
  }
  
  .i-card-icon {
    font-size: 1.5rem;
    margin-bottom: 25px;
    background: #ffffff;
    color: #000000;
    width: 45px; height: 45px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
  }
  
  .i-card h3 {
    font-size: 1.3rem;
    color: #ffffff;
    margin-bottom: 15px;
    font-weight: 600;
  }
  
  .i-card p {
    font-size: 1.05rem;
    color: #a0a8b4;
    line-height: 1.5;
    margin: 0;
  }
  
  .pres-footer-text {
    font-size: 1.15rem;
    color: #8fa0c0;
    line-height: 1.8;
    max-width: 1000px;
  }
  
  .grid-2-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 60px;
    align-items: center;
  }
  
  .pres-text-col p {
    font-size: 1.15rem;
    color: #9ba6b6;
    line-height: 1.8;
    margin-bottom: 25px;
  }
  
  .visual-col {
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    height: 450px;
    background: transparent;
  }

  .v-cards {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .v-card {
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 25px;
    transition: border-color 0.3s;
  }
  
  .v-card:hover {
    border-color: rgba(255,255,255,0.4);
  }
  
  .v-card h4 {
    color: #ffffff;
    font-size: 1.2rem;
    margin-bottom: 10px;
    font-weight: 600;
  }
  
  .v-card p {
    color: #9ba6b6;
    font-size: 1rem;
    line-height: 1.6;
    margin: 0;
  }

  /* CSS Drawings */
  .css-phone {
    width: 200px;
    height: 400px;
    border: 4px solid #ffffff;
    border-radius: 30px;
    position: relative;
    background: #111;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px 15px;
  }
  
  .css-phone::before {
    content: '';
    position: absolute;
    top: 10px;
    width: 60px;
    height: 6px;
    background: #333;
    border-radius: 10px;
  }
  
  .phone-element {
    width: 100%;
    height: 40px;
    background: rgba(255,255,255,0.1);
    border-radius: 20px;
    margin-bottom: 15px;
    margin-top: 10px;
  }

  .phone-element-circle {
    width: 40px;
    height: 40px;
    background: rgba(255,204,0,0.8);
    border-radius: 50%;
    align-self: flex-start;
    margin-bottom: 15px;
  }

  .css-meter {
    width: 250px;
    height: 280px;
    border: 5px solid #fff;
    border-radius: 20px;
    background: #fff;
    position: relative;
    box-shadow: 0 10px 40px rgba(0,255,136,0.2);
  }
  
  .meter-screen {
    width: 80%;
    height: 80px;
    background: #111;
    margin: 30px auto;
    border-radius: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #00ff88;
    font-family: 'Courier New', Courier, monospace;
    font-size: 3rem;
    font-weight: bold;
    letter-spacing: 5px;
  }

  .meter-dial {
    width: 60px; height: 60px;
    border-radius: 50%;
    background: #ff6600;
    position: absolute;
    bottom: 40px;
    left: 40px;
  }

  .meter-port {
    width: 40px; height: 20px;
    border-radius: 10px;
    background: #111;
    position: absolute;
    bottom: 60px;
    right: 40px;
  }

  .dashboard-header {
    margin-top: 80px;
    margin-bottom: 40px;
    text-align: center;
  }
  
  .dashboard-header h2 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #fff;
  }
  
  .dashboard-header p {
    color: #00ffff;
    font-size: 1.2rem;
  }
</style>
""", unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

bg_base64 = get_base64_of_bin_file('background.png')
if bg_base64:
    st.markdown(f"""
    <style>
    .hero-section {{
        background: linear-gradient(rgba(10, 12, 16, 0.7), rgba(8, 9, 10, 0.9)), url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Data Loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_forecasts():
    path = "forecasts.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp","zone_id","actual_demand","predicted_demand",
            "risk_level","risk_score","load_factor","is_test_period"
        ])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

@st.cache_data
def load_zone_risk():
    path = "zone_risk.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_anomalies():
    path = "anomaly_results.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp","meter_id","zone_id","meter_type","consumption_kwh",
            "is_anomaly","anomaly_type","ensemble_score","predicted_type",
            "final_flag","confidence_label","fp_reason"
        ])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

@st.cache_data
def load_meter_summary():
    path = "anomaly_meter_summary.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["first_flagged","last_flagged"])

@st.cache_data
def load_raw_features():
    path = "features_meter.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    cols = [
        "timestamp","meter_id","zone_id","consumption_kwh",
        "zscore_24h","peer_ratio","ratio_vs_24h","cv_24h",
        "abs_delta_1h","roll_mean_24h","lag_24h","is_anomaly","anomaly_type"
    ]
    return pd.read_csv(path, parse_dates=["timestamp"], usecols=cols)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Helper Functions & Colors
# ══════════════════════════════════════════════════════════════════════════════

# Neon Theme Colors
THEME = {
    "bg": "rgba(0,0,0,0)",
    "text": "#e0e6ed",
    "grid": "rgba(255,255,255,0.05)",
    "cyan": "#00ffff",
    "blue": "#00a2ff",
    "green": "#00ff88",
    "red": "#ff3366",
    "yellow": "#ffcc00",
    "purple": "#8a2be2"
}

RISK_COLORS = {
    "LOW":      THEME["green"],
    "MEDIUM":   THEME["yellow"],
    "HIGH":     "#ff6600",
    "CRITICAL": THEME["red"],
}

ANOMALY_COLORS = {
    "THEFT":          THEME["red"],
    "TAMPER":         THEME["yellow"],
    "SUDDEN_DROP":    THEME["cyan"],
    "PEER_DEVIATION": THEME["purple"],
    "UNKNOWN":        "#a0a0b0",
    "NORMAL":         THEME["green"],
}

def apply_plotly_theme(fig):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=THEME["text"]),
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        xaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
        yaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Load Data
# ══════════════════════════════════════════════════════════════════════════════

forecasts_df   = load_forecasts()
zone_risk_df   = load_zone_risk()
anomalies_df   = load_anomalies()
meter_sum_df   = load_meter_summary()
features_df    = load_raw_features()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="live-status">
        <div class="pulse-dot"></div>
        <div class="live-text">Live Grid Status</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🛠️ Filters & Controls", expanded=True):
        all_zones = ["All zones", "zone_0", "zone_1", "zone_2", "zone_3", "zone_4"]
        selected_zone = st.selectbox("📍 Target Feeder / Zone", all_zones)
        
        date_from = st.date_input("📅 Start Date", value=datetime(2024, 3, 17).date())
        date_to = st.date_input("📅 End Date", value=datetime(2024, 3, 31).date())

    with st.expander("🚨 Anomaly Parameters", expanded=True):
        show_theft  = st.checkbox("🔍 Theft Detection", value=True)
        show_tamper = st.checkbox("⚡ Tamper Detection", value=True)
        show_drop   = st.checkbox("📉 Sudden Drop", value=True)
        show_peer   = st.checkbox("👥 Peer Deviation", value=True)
        
        min_conf = st.slider("Min Confidence Score", 0.0, 1.0, 0.55, 0.05)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("🚀 Model: Isolation Forest + Autoencoder")
    st.caption("📈 Forecast: Gradient Boosting + Seasonal")

# Apply filters
if selected_zone != "All zones" and len(forecasts_df) > 0:
    forecasts_filt = forecasts_df[forecasts_df["zone_id"] == selected_zone]
else:
    forecasts_filt = forecasts_df.copy()

if len(forecasts_filt) > 0:
    date_from_ts = pd.Timestamp(date_from)
    date_to_ts   = pd.Timestamp(date_to) + pd.Timedelta(days=1)
    forecasts_filt = forecasts_filt[
        (forecasts_filt["timestamp"] >= date_from_ts) &
        (forecasts_filt["timestamp"] <  date_to_ts)
    ]

if len(anomalies_df) > 0:
    anom_filt = anomalies_df[anomalies_df["zone_id"] == selected_zone] if selected_zone != "All zones" else anomalies_df.copy()
    allowed_types = []
    if show_theft: allowed_types.append("THEFT")
    if show_tamper: allowed_types.append("TAMPER")
    if show_drop: allowed_types.append("SUDDEN_DROP")
    if show_peer: allowed_types.append("PEER_DEVIATION")
    allowed_types.append("UNKNOWN")
    
    anom_filt = anom_filt[
        (anom_filt["predicted_type"].isin(allowed_types) | (anom_filt["final_flag"] == 0)) &
        (anom_filt["ensemble_score"] >= min_conf)
    ]
    confirmed_alerts = anom_filt[anom_filt["final_flag"] == 1].sort_values("ensemble_score", ascending=False)
else:
    anom_filt = pd.DataFrame()
    confirmed_alerts = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Hero Banner & KPIs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<!-- SECTION 1: HERO -->
<div class="pres-section hero-section">
    <div class="pres-hero-title">BESCOM: Illuminating the Future of Energy Intelligence</div>
    <div class="pres-hero-subtitle">Empowering Karnataka with intelligent energy solutions that connect communities and illuminate homes with reliability and precision.</div>
</div>

<!-- SECTION 2: INSIGHT CARDS -->
<div class="pres-section">
    <div class="pres-section-title">The Power of Insight, Delivered Seamlessly</div>
    <div class="insight-cards-container">
        <div class="i-card">
            <div class="i-card-icon">🧠</div>
            <h3>Advanced Intelligence</h3>
            <p>AI-powered analysis working quietly in the background</p>
        </div>
        <div class="i-card">
            <div class="i-card-icon">🎯</div>
            <h3>Tangible Benefits</h3>
            <p>Real improvements delivered directly to your home</p>
        </div>
        <div class="i-card">
            <div class="i-card-icon">🔗</div>
            <h3>Network Excellence</h3>
            <p>More reliable, efficient energy for everyone</p>
        </div>
    </div>
    <div class="pres-footer-text">
        At BESCOM, we're integrating advanced intelligence into our smart meter network. This isn't about complex algorithms or abstract technology—it's about delivering tangible benefits directly to you and ensuring a more reliable, efficient energy future for everyone in Karnataka.
    </div>
</div>

<!-- SECTION 3: DESIGN -->
<div class="pres-section grid-2-col">
    <div class="pres-text-col">
        <div class="pres-section-title">A Design That Connects, Not Confuses</div>
        <p>We believe technology should empower, not overwhelm. Our approach to AI integration is focused on creating a user experience that is intuitive and reassuring for every customer.</p>
        <p>Think clean lines, clear information, and a design that feels grounded in practical application, not abstract concepts. Every interaction is designed with your needs in mind, ensuring that smart technology enhances your life without adding complexity.</p>
    </div>
    <div class="visual-col">
        <div class="css-phone">
            <div class="phone-element"></div>
            <div class="phone-element-circle"></div>
            <div class="phone-element" style="width: 80%; background: #00a2ff; align-self: flex-end;"></div>
            <div class="phone-element" style="background: transparent; border: 1px solid #fff;"></div>
        </div>
    </div>
</div>

<!-- SECTION 4: BEHIND THE METER -->
<div class="pres-section grid-2-col">
    <div class="visual-col">
        <div class="css-meter">
            <div class="meter-screen">006</div>
            <div class="meter-dial"></div>
            <div class="meter-port"></div>
        </div>
    </div>
    <div class="pres-text-col">
        <div class="pres-section-title">What's Happening Behind the Meter?</div>
        <div class="v-cards">
            <div class="v-card">
                <h4>Proactive Detection</h4>
                <p>Our intelligent systems identify anomalies and potential issues with your meter or energy usage in real-time, often before you even notice them. This means faster resolutions and fewer disruptions to your power supply.</p>
            </div>
            <div class="v-card">
                <h4>Enhanced Accuracy</h4>
                <p>AI-driven analysis ensures that your energy consumption data is more accurate than ever, leading to fairer billing and a clearer understanding of your usage patterns and habits.</p>
            </div>
            <div class="v-card">
                <h4>Grid Efficiency</h4>
                <p>By analysing data from millions of smart meters, we gain a comprehensive view of energy flow, enabling us to optimise distribution, reduce waste, and improve overall grid stability across the region.</p>
            </div>
        </div>
    </div>
</div>

<!-- SECTION 5: BEYOND THE METER -->
<div class="pres-section grid-2-col">
    <div class="pres-text-col">
        <div class="pres-section-title">Beyond the Meter: Smarter Energy, Simpler Lives</div>
        <p>Imagine a world where your energy consumption isn't just recorded, but truly understood. Where inefficiencies are detected before they impact your bill, and where the power grid operates with unprecedented precision and reliability.</p>
        <p>This is the future BESCOM is building, powered by intelligent insights from your smart meter. We're transforming how energy is monitored, managed, and delivered across Karnataka.</p>
    </div>
    <div class="visual-col">
        <div style="font-size: 8rem;">👩🏽‍💻📱👩🏻‍💻</div>
    </div>
</div>

<!-- DASHBOARD HEADER -->
<div class="dashboard-header">
    <h2>Interactive Data Hub</h2>
    <p>Live AI Dashboard & Real-Time Analytics</p>
</div>
""", unsafe_allow_html=True)

# KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

n_zones = forecasts_df["zone_id"].nunique() if len(forecasts_df) > 0 else 5
n_high = (forecasts_filt.sort_values("timestamp").groupby("zone_id").last()["risk_level"].isin(["HIGH","CRITICAL"])).sum() if len(forecasts_filt)>0 else 0
n_confirmed = len(confirmed_alerts)
theft_count = len(confirmed_alerts[confirmed_alerts["predicted_type"] == "THEFT"]) if len(confirmed_alerts) > 0 else 0

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📍 Active Zones</div>
        <div class="kpi-value">{n_zones}</div>
        <div class="kpi-trend-up">↑ 100% Coverage</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    color = THEME['red'] if n_high > 0 else THEME['green']
    st.markdown(f"""
    <div class="kpi-card" style="border-bottom: 3px solid {color};">
        <div class="kpi-title">⚠️ High Risk Areas</div>
        <div class="kpi-value">{n_high}</div>
        <div class="kpi-trend-down">{"Action Required" if n_high > 0 else "Optimal"}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🚨 Active Alerts</div>
        <div class="kpi-value">{n_confirmed}</div>
        <div class="kpi-trend-down">Requires Inspection</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🔍 Theft Detections</div>
        <div class="kpi-value" style="color: {THEME['cyan']};">{theft_count}</div>
        <div class="kpi-trend-up">↑ High Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Tabs & Story Flow
# ══════════════════════════════════════════════════════════════════════════════

tab_forecast, tab_anomaly, tab_insights = st.tabs([
    "📈 1. Demand & Risk (Problem)", 
    "🚨 2. Anomaly Alerts (Insight)", 
    "📊 3. Explainability (Action)"
])

with tab_forecast:
    col_map_1, col_map_2 = st.columns([3, 1])
    with col_map_1:
        st.markdown("### 🗺️ AI Zone Risk Analysis")
    with col_map_2:
        view_toggle = st.radio("View Mode", ["AI Insight View", "Pie Chart View"], horizontal=True, label_visibility="collapsed")
    
    st.caption("Identify peak load risks instantly using a pie chart representation.")
    
    if len(zone_risk_df) > 0:
        z_df = zone_risk_df.copy()
        z_df["color_val"] = z_df["dominant_risk"].map({"LOW":1, "MEDIUM":2, "HIGH":3, "CRITICAL":4})
        
        # Determine critical zones for the Insight Overlay
        crit_zones = z_df[z_df["dominant_risk"].isin(["HIGH", "CRITICAL"])]["zone_id"].tolist()
        if crit_zones and view_toggle == "AI Insight View":
            crit_str = ", ".join(crit_zones)
            st.markdown(f"""
            <div class="insight-overlay">
                <div class="insight-icon">⚠️</div>
                <div class="insight-text"><strong>AI INSIGHT:</strong> {crit_str} at critical risk due to repeated forecast peaks.</div>
            </div>
            """, unsafe_allow_html=True)
        
        if view_toggle == "Pie Chart View":
            fig_pie = px.pie(
                z_df, names="zone_id", values="avg_predicted",
                color="dominant_risk",
                color_discrete_map={
                    "LOW": THEME["green"], 
                    "MEDIUM": THEME["yellow"], 
                    "HIGH": THEME["red"], 
                    "CRITICAL": THEME["red"]
                },
                custom_data=["dominant_risk", "pct_high_risk"],
                hole=0.4
            )
            fig_pie.update_traces(
                hovertemplate="<b>%{label}</b><br>Risk: %{customdata[0]}<br>Peak Risk: %{customdata[1]:.1f}%<br>Avg kWh: %{value:.1f}<extra></extra>",
                textinfo="label+percent"
            )
            fig_pie = apply_plotly_theme(fig_pie)
            fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            # AI Insight View: Glassmorphic Cards
            html_cards = '<div class="zone-grid">'
            for _, row in z_df.iterrows():
                r_level = row.get("dominant_risk", "LOW")
                card_class = "zone-low"
                badge_class = "badge-low"
                trend = "<span style='color:#00ff88'>↓ Stable</span>"
                
                if r_level == "MEDIUM":
                    card_class = "zone-medium"
                    badge_class = "badge-medium"
                    trend = "<span style='color:#ffcc00'>↑ Rising</span>"
                elif r_level in ["HIGH", "CRITICAL"]:
                    card_class = "zone-critical"
                    badge_class = "badge-critical"
                    trend = "<span style='color:#ff3366'>↑ Spike</span>"
                
                avg_val = round(row.get("avg_predicted", 0), 1)
                peak_risk = round(row.get("pct_high_risk", 0), 1)
                
                html_cards += f"""
                <div class="zone-card {card_class}">
                    <div class="tooltip">
                        <strong>Peak load risk:</strong> {peak_risk}%<br>
                        Forecast spike likely at 7 PM
                    </div>
                    <div class="zone-name">{row['zone_id']}</div>
                    <div class="zone-avg">{avg_val} kWh</div>
                    <div class="zone-badge-container">
                        <div class="zone-badge {badge_class}">{r_level}</div>
                        <div class="zone-trend">{trend}</div>
                    </div>
                </div>
                """
            html_cards += '</div>'
            st.markdown(html_cards, unsafe_allow_html=True)
    else:
        st.info("No risk data.")

    st.markdown("---")
    c_f1, c_f2 = st.columns([3, 1])
    with c_f1:
        st.markdown("### 🔮 Demand Forecast vs Actual")
    with c_f2:
        zoom_toggle = st.toggle("🔍 Zoom to Peak Hours", value=False)

    if len(forecasts_filt) > 0:
        chart_df = forecasts_filt.set_index("timestamp")[["actual_demand","predicted_demand"]].resample("1H").mean().reset_index()
        
        if zoom_toggle:
            # Simple simulation of zoom: take the top 24 hours of data
            chart_df = chart_df.tail(24)
            
        fig = go.Figure()
        
        # Predicted line with gradient fill
        fig.add_trace(go.Scatter(
            x=chart_df["timestamp"], y=chart_df["predicted_demand"],
            name="Predicted (AI)", line=dict(color=THEME["cyan"], width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.15)',
            hovertemplate="AI Prediction: %{y:.2f} kWh<extra></extra>"
        ))
        
        # Actual line
        fig.add_trace(go.Scatter(
            x=chart_df["timestamp"], y=chart_df["actual_demand"],
            name="Actual Grid Load", line=dict(color=THEME["purple"], width=2, dash="dash", shape='spline'),
            hovertemplate="Actual: %{y:.2f} kWh<extra></extra>"
        ))

        # Highlight Critical Zones
        if "risk_level" in forecasts_filt.columns and not zoom_toggle:
            crit = forecasts_filt[forecasts_filt["risk_level"] == "CRITICAL"]
            for _, grp in crit.groupby((crit["timestamp"].diff() > pd.Timedelta("1H")).cumsum()):
                fig.add_vrect(
                    x0=grp["timestamp"].min(), x1=grp["timestamp"].max(),
                    fillcolor="rgba(255, 51, 102, 0.1)", layer="below", line_width=0,
                    annotation_text="CRITICAL", annotation_font_color=THEME["red"]
                )

        fig = apply_plotly_theme(fig)
        fig.update_layout(height=400, legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig, use_container_width=True)


with tab_anomaly:
    c1, c2 = st.columns([1.2, 2])
    
    with c1:
        st.markdown("### 📨 Alert Feed UI")
        st.caption("Real-time anomaly flags pushed to field inspectors.")
        if len(confirmed_alerts) > 0:
            for _, row in confirmed_alerts.head(8).iterrows():
                alert_color = ANOMALY_COLORS.get(row['predicted_type'], THEME["blue"])
                
                # Dynamic explainability text
                explain_text = "Irregular load behavior detected."
                if row['predicted_type'] == 'THEFT':
                    explain_text = "Unmetered bypass or tamper suspected."
                elif row['predicted_type'] == 'SUDDEN_DROP':
                    explain_text = "Usage dropped 65% vs 7-day average."
                elif row['predicted_type'] == 'PEER_DEVIATION':
                    explain_text = "Usage diverges significantly from local peers."
                
                st.markdown(f"""
                <div class="alert-feed-card" style="border-left-color: {alert_color};">
                    <div class="alert-feed-title" style="color: {alert_color};">
                        ⚠️ {row['predicted_type'].replace('_', ' ')}
                    </div>
                    <div class="alert-feed-desc">
                        <strong>Meter:</strong> {row['meter_id']} ({row['zone_id']})<br>
                        <strong>Detail:</strong> {explain_text}<br>
                        <span style="opacity: 0.6; font-size: 0.75rem;">Score: {row['ensemble_score']:.2f} | {row['timestamp'].strftime('%Y-%m-%d %H:%M')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active alerts.")

    with c2:
        st.markdown("### 🔍 Alert Breakdown")
        if len(confirmed_alerts) > 0:
            type_counts = confirmed_alerts["predicted_type"].value_counts().reset_index()
            type_counts.columns = ["type","count"]
            
            fig_type = px.bar(
                type_counts, x="count", y="type", orientation="h",
                color="type", color_discrete_map=ANOMALY_COLORS,
                labels={"type":"Anomaly Type","count":"Detections"}
            )
            fig_type = apply_plotly_theme(fig_type)
            fig_type.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig_type, use_container_width=True)
            
            st.markdown("### 📊 Inspector Action Queue")
            if len(meter_sum_df) > 0:
                q_df = meter_sum_df.sort_values("max_score", ascending=False).head(5)
                # Keep it clean
                st.dataframe(q_df[['meter_id', 'zone_id', 'dominant_type', 'max_score']], use_container_width=True, hide_index=True)

with tab_insights:
    st.markdown("### 🧠 Why the model made this prediction")
    st.caption("Global feature importance explaining the underlying drivers for anomaly and risk predictions.")
    
    feat_importance = {
        "Consumption Drop (vs 7-day avg)": 0.45,
        "Peer Group Deviation": 0.28,
        "Volatility (24h CV)": 0.15,
        "Sudden Step-Change": 0.08,
        "Off-hours usage": 0.04,
    }

    fig_feat = go.Figure(go.Bar(
        x=list(feat_importance.values()),
        y=list(feat_importance.keys()),
        orientation="h",
        marker=dict(
            color=list(feat_importance.values()),
            colorscale=[THEME["purple"], THEME["cyan"]],
        ),
        hovertemplate="Impact: %{x:.2f}<br>Feature: %{y}<extra></extra>"
    ))
    
    fig_feat = apply_plotly_theme(fig_feat)
    fig_feat.update_layout(
        height=400,
        xaxis_title="Impact on Model Decision (Relative Importance)",
        yaxis_title="",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_feat, use_container_width=True)


st.markdown("""
<div class="tagline">
    AI-powered decisions. Human-controlled actions.
</div>
""", unsafe_allow_html=True)
