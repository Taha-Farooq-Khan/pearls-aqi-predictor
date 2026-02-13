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

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multan AQI Predictor | MLOps",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Custom CSS for Polish
st.markdown("""
    <style>
    /* Main typography and spacing */
    .main-header {font-size: 2.8rem; font-weight: 800; color: #1E1E1E; padding-bottom: 0px; margin-bottom: 0px;}
    .sub-header {font-size: 1.1rem; color: #666; font-weight: 400; margin-bottom: 2rem;}

    /* Button Styling */
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        background-color: #2E3B55; 
        color: white; 
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1a233a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Metric Cards Styling */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #f0f2f6;
        height: 100%;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Prediction Cards Styling */
    .day-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #ccc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .day-card h4 { margin-top: 0; color: #333; font-size: 1.1rem; }
    .day-card h2 { margin: 10px 0; font-size: 1.8rem; }
    .day-card p { margin-bottom: 0; font-size: 0.9rem; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

# AUTHENTICATION
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

try:
    API_KEY = st.secrets["HOPSWORKS_API_KEY"]
except:
    API_KEY = os.getenv('HOPSWORKS_API_KEY')



# HELPER FUNCTIONS
def get_aqi_category(pm25):
    """Returns Category, Hex Color, and Emoji based on PM2.5"""
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
    return hopsworks.login(api_key_value=API_KEY)


def forecast_recursive(model, df_recent, scaler, feature_cols, model_type="deep_learning", hours_to_predict=168):
    """
    Advanced recursive forecasting for 7 Days (168 Hours).
    """
    history_pm25 = df_recent['pm25'].values[-24:].tolist()
    last_weather = df_recent.iloc[-1][['temp', 'humidity', 'wind_speed', 'rain']].to_dict()
    current_time = pd.to_datetime(df_recent.iloc[-1]['date'])
    predictions = []

    for i in range(hours_to_predict):
        next_time = current_time + timedelta(hours=i + 1)
        h_sin = np.sin(2 * np.pi * next_time.hour / 24)
        h_cos = np.cos(2 * np.pi * next_time.hour / 24)

        lag_1 = history_pm25[-1]
        lag_6 = history_pm25[-6]
        lag_24 = history_pm25[-24]
        roll_mean = np.mean(history_pm25[-24:])
        interaction = last_weather['humidity'] * last_weather['temp']

        row_raw = pd.DataFrame([[
            lag_1, lag_6, lag_24, roll_mean,
            last_weather['temp'], last_weather['humidity'], last_weather['wind_speed'], last_weather['rain'],
            h_sin, h_cos, interaction
        ]], columns=feature_cols)

        row_scaled = scaler.transform(row_raw)

        if model_type == "deep_learning":
            model_input = row_scaled.reshape(1, 1, -1)
        else:
            model_input = row_scaled

        try:
            if model_type == "deep_learning":
                pred_real = model.predict(model_input, verbose=0)[0][0]
            else:
                pred_real = model.predict(model_input)[0]
        except:
            pred_real = history_pm25[-1]

        pred_real = max(0, pred_real)  # Safety clip
        predictions.append(pred_real)
        history_pm25.append(pred_real)

    return predictions



# MAIN UI

# Header Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="main-header">Multan AQI Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Air Quality Forecasting System | <b>Live MLOps Pipeline</b></div>',
                unsafe_allow_html=True)
with col_head2:
    run_button = st.button("🔮 Generate Live Forecast", use_container_width=True)

st.divider()

if run_button:
    status_box = st.status("Initializing MLOps Pipeline...", expanded=True)

    try:
        # 1. CONNECT
        status_box.write("🔌 Authenticating with Hopsworks Feature Store...")
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        mr = project.get_model_registry()

        # 2. DETERMINE CHAMPION MODEL
        status_box.write("🏆 Querying Model Registry for Champion Model...")
        model_names = ["lightgbm", "catboost", "lstm", "cnn"]
        best_rmse, best_model_name = float('inf'), "cnn"
        metrics_display = []

        for name in model_names:
            try:
                m = mr.get_model(f"aqi_{name}_multan", version=None)
                rmse = m.training_metrics.get('rmse', 999)
                mae = m.training_metrics.get('mae', 0)  # <-- FETCH MAE HERE
                r2 = m.training_metrics.get('r2', 0)

                # <-- ADD MAE TO THE DISPLAY DICTIONARY
                metrics_display.append({"Model": name.upper(), "RMSE": rmse, "MAE": mae, "R²": r2})

                if rmse < best_rmse:
                    best_rmse, best_model_name = rmse, name
            except:
                pass

        status_box.write(f"✅ Active Champion: **{best_model_name.upper()}** (RMSE: {best_rmse:.2f})")

        # 3. LOAD CHAMPION
        status_box.write("🧠 Downloading model weights into memory...")
        retrieved_model = mr.get_model(f"aqi_{best_model_name}_multan", version=None)
        saved_dir = retrieved_model.download()

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
        status_box.write("📡 Fetching real-time telemetry from Feature Store...")
        aqi_fg = fs.get_feature_group(name="aqi_data_multan", version=1)
        df = aqi_fg.read().sort_values(by="timestamp")
        latest_row = df.iloc[-1]

        features = ['pm25_lag_1', 'pm25_lag_6', 'pm25_lag_24', 'pm25_rolling_mean_24h',
                    'temp', 'humidity', 'wind_speed', 'rain', 'hour_sin', 'hour_cos', 'humid_temp_interaction']

        scaler = MinMaxScaler().fit(df[features])

        # 5. RUN 7-DAY FORECAST
        status_box.write("📈 Computing 168-Hour Recursive Predictions...")
        df_recent = df.tail(30)
        # Predict 168 hours (7 days)
        preds_real = forecast_recursive(model, df_recent, scaler, features, model_type, hours_to_predict=168)

        # Build Forecast DataFrame
        current_date_dt = pd.to_datetime(latest_row['date'])
        future_dates = [current_date_dt + timedelta(hours=i + 1) for i in range(168)]
        forecast_df = pd.DataFrame({'Date': future_dates, 'PM2.5': preds_real})

        status_box.update(label="✨ Analysis Complete. Rendering Dashboard.", state="complete", expanded=False)


        # DASHBOARD RENDERING

        # --- SECTION 1: CURRENT CONDITIONS ---
        st.subheader("📍 Current Air Quality Conditions")
        curr_pm = latest_row['pm25']
        cat, color, emoji = get_aqi_category(curr_pm)

        col_main1, col_main2, col_main3 = st.columns([1.5, 2, 1])

        with col_main1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 5px solid {color};">
                <div class="metric-label">PM2.5 Concentration</div>
                <div class="metric-value" style="color: {color}; font-size: 3.5rem;">{curr_pm:.0f}</div>
                <div style="font-size: 1.2rem; font-weight: 600;">{emoji} {cat}</div>
                <div style="color: #888; font-size: 0.8rem; margin-top: 10px;">Last Updated: {current_date_dt.strftime('%H:%M %p')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_main2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=curr_pm,
                gauge={
                    'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [0, 50], 'color': "#00e400"},
                        {'range': [50, 100], 'color': "#ffff00"},
                        {'range': [100, 150], 'color': "#ff7e00"},
                        {'range': [150, 200], 'color': "#ff0000"},
                        {'range': [200, 500], 'color': "#7e0023"}],
                    'threshold': {'line': {'color': "#1E1E1E", 'width': 6}, 'thickness': 0.8, 'value': curr_pm}
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_main3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Weather Context</div>
                <div class="metric-value" style="font-size: 1.8rem;">{latest_row['temp']:.1f}°C</div>
                <div style="color: #666;">Temp</div>
                <hr style="margin: 10px 0;">
                <div class="metric-value" style="font-size: 1.8rem;">{latest_row['humidity']:.0f}%</div>
                <div style="color: #666;">Humidity</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")  # Spacer

        # --- SECTION 2: NEXT 3 DAYS SUMMARY CARDS ---
        st.subheader("📅 3-Day Outlook Summary")

        # Calculate daily averages from the forecast dataframe
        daily_forecasts = []
        for day_offset in range(1, 4):
            target_date = current_date_dt.date() + timedelta(days=day_offset)
            day_data = forecast_df[forecast_df['Date'].dt.date == target_date]
            if not day_data.empty:
                avg_pm25 = day_data['PM2.5'].mean()
                daily_forecasts.append({"date": target_date, "pm25": avg_pm25})

        col_d1, col_d2, col_d3 = st.columns(3)
        cols = [col_d1, col_d2, col_d3]

        for idx, day_data in enumerate(daily_forecasts):
            with cols[idx]:
                day_name = day_data["date"].strftime("%A")
                date_str = day_data["date"].strftime("%b %d")
                d_cat, d_color, d_emoji = get_aqi_category(day_data["pm25"])

                st.markdown(f"""
                <div class="day-card" style="border-left-color: {d_color};">
                    <h4>{day_name}</h4>
                    <div style="color: #777; font-size: 0.8rem;">{date_str}</div>
                    <h2 style="color: {d_color};">{day_data["pm25"]:.0f}</h2>
                    <p>{d_emoji} {d_cat}</p>
                </div>
                """, unsafe_allow_html=True)

        # --- SECTION 3: 7-DAY HOURLY TREND CHART ---
        st.divider()
        st.subheader("📈 Full 7-Day Hourly Trend")

        fig = px.area(forecast_df, x='Date', y='PM2.5',
                      labels={"PM2.5": "PM2.5 Concentration (µg/m³)", "Date": ""})

        # Styling to make it look professional
        fig.update_traces(
            line_color='#2E3B55',
            fill='tozeroy',
            fillcolor='rgba(46, 59, 85, 0.1)'
        )

        # Safe limits and hazardous lines
        fig.add_hline(y=35, line_dash="dash", line_color="#00e400", annotation_text=" WHO Safe Limit (35) ",
                      annotation_position="bottom right")
        fig.add_hline(y=150, line_dash="dash", line_color="#ff0000", annotation_text=" Unhealthy (150) ",
                      annotation_position="top right")

        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', gridwidth=1),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- SECTION 4: MLOps METRICS ---
        with st.expander("🛠️ View Pipeline Metrics & Model Leaderboard"):
            st.caption("Live comparison of algorithms tracked by the Hopsworks Model Registry.")
            metrics_df = pd.DataFrame(metrics_display).sort_values(by="RMSE")
            st.dataframe(
                metrics_df.style.format({"RMSE": "{:.2f}", "R²": "{:.4f}"}),
                hide_index=True,
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Pipeline Error: {e}")

else:
    st.info("👆 Click **'Generate Live Forecast'** to trigger the MLOps pipeline and load the dashboard.")