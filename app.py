import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="GreenShield AI - Disaster Prediction",
    page_icon="🛡️",
    layout="wide"
)

# --- File Paths ---
# FLOOD_CSV_PATH = r"C:\\Users\\itssj\\OneDrive\\Desktop\\Programming\\AI_ML\\flood.csv"
# WILDFIRE_CSV_PATH = r"C:\\Users\\itssj\\OneDrive\\Desktop\\Programming\\AI_ML\\CA_Weather_Fire_Dataset_1984-2025.csv"
FLOOD_CSV_PATH = "flood.csv"
WILDFIRE_CSV_PATH = "CA_Weather_Fire_Dataset_1984-2025.csv"

# --- Cache Data Loaders ---
@st.cache_data
def load_flood_data():
    df = pd.read_csv(FLOOD_CSV_PATH)
    if 'FloodProbability' in df.columns:
        df['flood_occurred'] = np.where(df['FloodProbability'] > 0.5, 1, 0)
        df = df.drop(columns=['FloodProbability'])
    else:
        df['flood_occurred'] = np.random.randint(0, 2, size=len(df))
    df = df.dropna()
    return df

@st.cache_data
def load_wildfire_data():
    df = pd.read_csv(WILDFIRE_CSV_PATH)
    drop_cols = ['Date', 'Location', 'County', 'State', 'Region',
                 'Latitude', 'Longitude', 'System:index', 'time', '.geo']
    df = df.drop(columns=drop_cols, errors='ignore')
    if 'labels' in df.columns:
        df = df.rename(columns={'labels': 'wildfire_occurred'})
    else:
        df['wildfire_occurred'] = np.random.randint(0, 2, size=len(df))
    df = df.select_dtypes(include=[np.number]).dropna()
    return df

# --- Cache Model Training ---
@st.cache_resource
def train_model(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 40px;
        background-color: #0E1117;
        border-radius: 15px;
        margin-bottom: 10px;
    }
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #00FF88;
        text-shadow: 0 0 10px #00FF88;
        margin: 0;
    }
    .toggle-container {
        position: relative;
        width: 160px;
        height: 55px;
        background: linear-gradient(90deg, #00bfff, #ff8c00);
        border-radius: 50px;
        display: flex;
        align-items: center;
        padding: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.3);
    }
    .thumb {
        width: 70px;
        height: 45px;
        background-color: white;
        border-radius: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        transition: all 0.3s ease-in-out;
        position: absolute;
        top: 5px;
        left: 5px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .thumb.right {
        left: 85px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Persistent State ---
if "is_wildfire" not in st.session_state:
    st.session_state.is_wildfire = False

# --- Header with Title + Toggle on Top Right ---
st.markdown(f"""
<div class="main-header">
    <div class="main-title">🛡️ GreenShield AI - Disaster Predictor</div>
    <div class="toggle-container">
        <div class="thumb {'right' if st.session_state.is_wildfire else ''}">
            {'🔥' if st.session_state.is_wildfire else '🌊'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Small Switch Button Below Toggle ---
col1, col2, col3 = st.columns([3, 2, 1])
with col3:  # place on right
    if st.button("Switch"):
        st.session_state.is_wildfire = not st.session_state.is_wildfire
        st.rerun()

# --- Mode Logic ---
if not st.session_state.is_wildfire:
    df = load_flood_data()
    target_col = 'flood_occurred'
    title = "🌊 Flood Risk Prediction"
    header_color = "#00FFFF"
else:
    df = load_wildfire_data()
    target_col = 'wildfire_occurred'
    title = "🔥 Wildfire Risk Prediction"
    header_color = "#FF8000"

# --- Train Model ---
model = train_model(df, target_col)
feature_names = df.columns.drop(target_col)

# --- Section Header ---
st.markdown(f"""
<div style="text-align:center; padding:20px; background-color:#0E1117; border-radius:15px;">
    <h1 style='color:{header_color};'>{title}</h1>
    <p style='color:#ccc;'>Adjust environmental factors below to predict the risk level.</p>
</div>
""", unsafe_allow_html=True)

# --- Sliders for Input ---
cols = st.columns(3)
features = {}
for i, feature in enumerate(feature_names):
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    features[feature] = cols[i % 3].slider(feature.replace("_", " "), min_val, max_val, mean_val)

input_df = pd.DataFrame([features])

# --- Prediction Button with Risk Box + Progress Bar + Contextual Message Below ---
if st.button(f"🔮 Predict {title.split()[1]} Risk"):
    prediction = model.predict(input_df)
    proba_output = model.predict_proba(input_df)
    risk_percentage = proba_output[:, 1][0] * 100 if proba_output.shape[1] > 1 else 0.0

    # --- Risk Box first ---
    if prediction[0] == 1:
        st.markdown(f"""
            <div style='padding:20px; border-radius:15px; background-color:#ff4b4b; color:white'>
            ❗ <b>HIGH {title.split()[1].upper()} RISK</b> - Likelihood: <b>{risk_percentage:.2f}%</b>
            </div>
        """, unsafe_allow_html=True)
        # --- Progress bar below risk box ---
        st.progress(risk_percentage / 100)
        # --- Contextual message below bar ---
        st.markdown("⚠️ Caution! Risk is high—take necessary precautions.")
    else:
        st.markdown(f"""
            <div style='padding:20px; border-radius:15px; background-color:#4CAF50; color:white'>
            ✅ <b>LOW {title.split()[1].upper()} RISK</b> - Likelihood: <b>{risk_percentage:.2f}%</b>
            </div>
        """, unsafe_allow_html=True)
        # --- Progress bar below risk box ---
        st.progress(risk_percentage / 100)
        # --- Contextual message below bar ---
        st.markdown("🎉 You’re safe for now! Keep monitoring conditions.")
