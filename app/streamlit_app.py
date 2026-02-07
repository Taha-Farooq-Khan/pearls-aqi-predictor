import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from pathlib import Path
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=UserWarning)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Multan AQI Prediction (Pearls AQI)",
    page_icon="🌫️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 40px; font-weight: bold; color: #1E1E1E;}
    .sub-header {font-size: 18px; color: #555;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# ───────────────────────────────
# AUTHENTICATION

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv('HOPSWORKS_API_KEY')


# HELPER FUNCTIONS
def get_aqi_category(pm25):
    if pm25 <= 50:
        return "Good", "#00e400", "🟢"
    elif pm25 <= 100:
        return "Satisfactory", "#ffff00", "🟡"
    elif pm25 <= 150:
        return "Moderate", "#ff7e00", "🟠"
    elif pm25 <= 200:
        return "Poor", "#ff0000", "🔴"
    elif pm25 <= 300:
        return "Very Poor", "#8f3f97", "🟣"
    else:
        return "Hazardous", "#7e0023", "🟤"


@st.cache_resource
def get_hopsworks_project():
    if not API_KEY: return None
    project = hopsworks.login(api_key_value=API_KEY)
    return project


def forecast_recursive(model, df_recent, scaler, model_type="deep_learning"):
    """
    Advanced recursive forecasting that updates Time and Lags correctly.
    """
    # Define exact feature order used in training
    feature_cols = ['pm25_lag_1', 'pm25_lag_6', 'pm25_lag_24', 'pm25_rolling_mean_24h',
                    'temp', 'humidity', 'wind_speed', 'rain', 'hour_sin', 'hour_cos', 'humid_temp_interaction']

    # 1. Prepare History Buffer
    history_pm25 = df_recent['pm25'].values[-24:].tolist()
    last_weather = df_recent.iloc[-1][['temp', 'humidity', 'wind_speed', 'rain']].to_dict()
    current_time = pd.to_datetime(df_recent.iloc[-1]['date'])

    predictions = []

    for i in range(72):
        # A. Update Time
        next_time = current_time + timedelta(hours=i + 1)
        h_sin = np.sin(2 * np.pi * next_time.hour / 24)
        h_cos = np.cos(2 * np.pi * next_time.hour / 24)

        # B. Calculate Lags
        lag_1 = history_pm25[-1]
        lag_6 = history_pm25[-6]
        lag_24 = history_pm25[-24]
        roll_mean = np.mean(history_pm25[-24:])

        interaction = last_weather['humidity'] * last_weather['temp']

        # C. Build Feature Vector
        row_raw = np.array([[
            lag_1, lag_6, lag_24, roll_mean,
            last_weather['temp'], last_weather['humidity'], last_weather['wind_speed'], last_weather['rain'],
            h_sin, h_cos, interaction
        ]])

        # FIX: Convert to DataFrame with columns
        row_df = pd.DataFrame(row_raw, columns=feature_cols)

        # D. Scale & Reshape
        row_scaled = scaler.transform(row_df)

        if model_type == "deep_learning":
            model_input = row_scaled.reshape(1, 1, -1)
        else:
            model_input = row_scaled

        # E. Predict
        try:
            if model_type == "deep_learning":
                pred_scaled = model.predict(model_input, verbose=0)[0][0]
            else:
                pred_scaled = model.predict(model_input)[0]
        except:
            pred_scaled = row_scaled[0][0]  # Fallback

        # Unscale
        pm_col_idx = 0
        pm_min = scaler.data_min_[pm_col_idx]
        pm_max = scaler.data_max_[pm_col_idx]
        pred_real = pred_scaled * (pm_max - pm_min) + pm_min

        predictions.append(pred_real)
        history_pm25.append(pred_real)

    return predictions


# MAIN UI

st.markdown('<div class="main-header">Multan AQI Prediction (Pearls AQI)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated MLOps Pipeline | Data source: <b>Hopsworks Feature Store</b></div>',
            unsafe_allow_html=True)
st.divider()

if st.button("🚀 Make Prediction"):

    status_box = st.status("Connecting to MLOps Pipeline...", expanded=True)

    try:
        # 1. CONNECT
        status_box.write("🔌 Authenticating with Hopsworks...")
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        mr = project.get_model_registry()

        # 2. DETERMINE CHAMPION MODEL (Updated to include CNN)
        status_box.write("🏆 Checking Model Registry for the Best Model...")
        model_names = ["lightgbm", "catboost", "lstm", "cnn"]
        best_rmse = float('inf')
        best_model_name = "lstm"
        metrics_display = []

        for name in model_names:
            try:
                m = mr.get_model(f"aqi_{name}_multan", version=1)
                rmse = m.training_metrics.get('rmse', 999)
                r2 = m.training_metrics.get('r2', 0)

                metrics_display.append({
                    "Model": name.upper(),
                    "RMSE (Error)": rmse,
                    "R² Score (Accuracy)": r2
                })

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = name
            except:
                pass

        status_box.write(f"✅ Champion Found: **{best_model_name.upper()}** (RMSE: {best_rmse:.4f})")

        # 3. LOAD CHAMPION
        status_box.write(f"🧠 Loading {best_model_name.upper()} into memory...")
        retrieved_model = mr.get_model(f"aqi_{best_model_name}_multan", version=1)
        saved_dir = retrieved_model.download()

        # Determine Model Type (Deep Learning vs Trees)
        # Note: CNN is also Deep Learning, so we group it with LSTM
        if best_model_name in ["lstm", "cnn"]:
            import tensorflow as tf

            model_path = glob.glob(f"{saved_dir}/*.keras")[0]
            model = tf.keras.models.load_model(model_path)
            model_type = "deep_learning"
        else:
            model_path = glob.glob(f"{saved_dir}/*.pkl")[0]
            model = joblib.load(model_path)
            model_type = "sklearn"

        # 4. PREPARE DATA
        status_box.write("📡 Fetching latest data from Feature Store...")
        aqi_fg = fs.get_feature_group(name="aqi_data_multan", version=1)
        df = aqi_fg.read().sort_values(by="timestamp")
        latest_row = df.iloc[-1]

        features = ['pm25_lag_1', 'pm25_lag_6', 'pm25_lag_24', 'pm25_rolling_mean_24h',
                    'temp', 'humidity', 'wind_speed', 'rain', 'hour_sin', 'hour_cos', 'humid_temp_interaction']
        scaler = MinMaxScaler()
        scaler.fit(df[features])

        # 5. RUN FORECAST
        status_box.write("🔮 Running 72-Hour Recursive Forecast...")
        df_recent = df.tail(30)
        preds_real = forecast_recursive(model, df_recent, scaler, model_type)

        status_box.update(label="✨ Success! Prediction Ready.", state="complete", expanded=False)

        # DISPLAY RESULTS

        # A. Current Gauge
        st.subheader("📍 Current Air Quality")
        col1, col2 = st.columns([1, 2])
        curr_pm = latest_row['pm25']
        cat, color, emoji = get_aqi_category(curr_pm)

        with col1:
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: black; text-align: center;">
                <h2 style="margin:0; font-size: 50px;">{curr_pm:.0f}</h2>
                <p style="margin:0; font-weight: bold;">Current PM2.5</p>
                <h3 style="margin:0;">{emoji} {cat}</h3>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=curr_pm,
                title={'text': "PM2.5 Indicator"},
                gauge={
                    'axis': {'range': [0, 500]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [0, 50], 'color': "#00e400"},
                        {'range': [50, 100], 'color': "#ffff00"},
                        {'range': [100, 150], 'color': "#ff7e00"},
                        {'range': [150, 200], 'color': "#ff0000"},
                        {'range': [200, 500], 'color': "#7e0023"}],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': curr_pm}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # B. 72-Hour Chart
        st.divider()
        st.subheader(f"🔮 72-Hour Forecast (Model: {best_model_name.upper()})")

        future_dates = [pd.to_datetime(latest_row['date']) + timedelta(hours=i + 1) for i in range(72)]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted PM2.5': preds_real})

        fig_line = px.line(forecast_df, x='Date', y='Predicted PM2.5')
        fig_line.update_traces(line_color='#FF4B4B', line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)

        # C. Model Leaderboard
        st.divider()
        st.subheader("🏆 Model Leaderboard")
        st.caption("Lower RMSE is better (less error). Higher R² is better (more accuracy).")

        metrics_df = pd.DataFrame(metrics_display).sort_values(by="RMSE (Error)")

        st.dataframe(
            metrics_df.style.format({
                "RMSE (Error)": "{:.4f}",
                "R² Score (Accuracy)": "{:.4f}"
            }),
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👆 Click **'Make Prediction'** to start.")