import pandas as pd
import requests
import hopsworks
import os
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv


load_dotenv()
try:
    API_KEY = os.getenv('HOPSWORKS_API_KEY')
except:
    raise ValueError("ERROR!: API Key not found! Check your .env file.")

# 2. Connect to Hopsworks
print("Connecting to Hopsworks Feature Store...")
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()

# FETCH DATA
LAT = 30.1968
LON = 71.4782
end_date = datetime.now()
start_date = end_date - timedelta(days=365) # Let's fetch a full year for safety

print(f" Fetching 1 Year of Data for Multan ({start_date.date()} to {end_date.date()})...")

# A. Weather Data
weather_url = "https://archive-api.open-meteo.com/v1/archive"
params_w = {
    "latitude": LAT, "longitude": LON,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain",
    "timezone": "auto"
}
response_w = requests.get(weather_url, params=params_w).json()
df_w = pd.DataFrame(response_w['hourly'])

# B. Pollution Data
air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params_a = {
    "latitude": LAT, "longitude": LON,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": "pm10,pm2_5",
    "timezone": "auto"
}
response_a = requests.get(air_url, params=params_a).json()
df_a = pd.DataFrame(response_a['hourly'])

# CLEAN & MERGE
# Rename
df_w.rename(columns={'time': 'date', 'temperature_2m': 'temp',
                     'relative_humidity_2m': 'humidity', 'wind_speed_10m': 'wind_speed'}, inplace=True)
df_a.rename(columns={'time': 'date', 'pm2_5': 'pm25'}, inplace=True)

# Merge
df_w['date'] = pd.to_datetime(df_w['date'])
df_a['date'] = pd.to_datetime(df_a['date'])
df = pd.merge(df_w, df_a, on='date')

# FEATURE ENGINEERING
print("Engineering Features...")

# A. Lags
df['pm25_lag_1'] = df['pm25'].shift(1)
df['pm25_lag_6'] = df['pm25'].shift(6)
df['pm25_lag_24'] = df['pm25'].shift(24)

# B. Rolling Stats
df['pm25_rolling_mean_24h'] = df['pm25'].rolling(window=24).mean()
df['pm25_rolling_std_24h'] = df['pm25'].rolling(window=24).std()

# C. Cyclical Time
df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)

# D. Interaction
df['humid_temp_interaction'] = df['humidity'] * df['temp']

# Hopsworks needs a "timestamp" column (integer milliseconds) for the index
# df['timestamp'] = df['date'].values.astype(int) // 10**6
# FIX: Force timestamp to be int64 (Big Integer) for Hopsworks
df['timestamp'] = (df['date'].values.astype("int64") // 10**6).astype("int64")

# Drop NaNs (created by lags)
df = df.dropna()

print(f"Data Prepared: {len(df)} rows.")

# --- STEP 6: UPLOAD TO FEATURE STORE ---
print("Uploading to Cloud...")

# Create the Feature Group (Table)
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_data_multan",
    version=1,
    primary_key=["timestamp"],
    event_time="timestamp",
    description="Multan AQI Data (Open-Meteo) with Lags"
)

# Insert Data
aqi_fg.insert(df)
print("Success! Check Hopsworks Dashboard.")