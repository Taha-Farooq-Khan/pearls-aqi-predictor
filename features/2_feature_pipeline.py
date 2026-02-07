import pandas as pd
import requests
import hopsworks
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# AUTHENTICATION
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv('HOPSWORKS_API_KEY')
if not API_KEY:
    raise ValueError("ERROR! API Key not found! Check your .env file.")

# CONFIGURATION
LAT = 30.1968
LON = 71.4782

# CONNECT TO HOPSWORKS
print("Connecting to Feature Store...")
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()

# Connect to the EXISTING Feature Group
aqi_fg = fs.get_feature_group(name="aqi_data_multan", version=1)

# FETCH ONLY RECENT DATA
# We fetch a 3-day window to ensure we can calculate "Lags" (24h) correctly
end_date = datetime.now()
start_date = end_date - timedelta(days=3)

print(f"Fetching fresh data ({start_date.date()} to {end_date.date()})...")

# --- CRITICAL FIX: Use 'api.open-meteo.com' (Forecast) instead of 'archive' for recent data ---
weather_url = "https://api.open-meteo.com/v1/forecast"
params_w = {
    "latitude": LAT, "longitude": LON,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain",
    "timezone": "auto"
}
response_w = requests.get(weather_url, params=params_w)

# Check if API call was successful
if response_w.status_code != 200:
    raise Exception(f"Weather API Error: {response_w.text}")

df_w = pd.DataFrame(response_w.json()['hourly'])

# B. Pollution (Air Quality API is fine for both history and forecast)
air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params_a = {
    "latitude": LAT, "longitude": LON,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": "pm10,pm2_5",
    "timezone": "auto"
}
response_a = requests.get(air_url, params=params_a)

if response_a.status_code != 200:
    raise Exception(f"Pollution API Error: {response_a.text}")

df_a = pd.DataFrame(response_a.json()['hourly'])

# PROCESSING
# Rename
df_w.rename(columns={'time': 'date', 'temperature_2m': 'temp',
                     'relative_humidity_2m': 'humidity', 'wind_speed_10m': 'wind_speed'}, inplace=True)
df_a.rename(columns={'time': 'date', 'pm2_5': 'pm25'}, inplace=True)

# Merge
df_w['date'] = pd.to_datetime(df_w['date'])
df_a['date'] = pd.to_datetime(df_a['date'])
df = pd.merge(df_w, df_a, on='date')

# Engineer Features
df['pm25_lag_1'] = df['pm25'].shift(1)
df['pm25_lag_6'] = df['pm25'].shift(6)
df['pm25_lag_24'] = df['pm25'].shift(24)
df['pm25_rolling_mean_24h'] = df['pm25'].rolling(window=24).mean()
df['pm25_rolling_std_24h'] = df['pm25'].rolling(window=24).std()
df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
df['humid_temp_interaction'] = df['humidity'] * df['temp']

# TIMESTAMPS: Use the BigInt Fix (Crucial!)
df['timestamp'] = (df['date'].values.astype("int64") // 10**6).astype("int64")

# Drop NaNs (This will drop the first 24 hours of our 3-day window, which is expected)
df = df.dropna()

print(f"New Data Processed: {len(df)} rows.")

if len(df) > 0:
    # UPLOAD
    print("Pushing to Cloud...")
    # Hopsworks automatically handles duplicates using the Primary Key (timestamp)
    # It will update existing rows and insert new ones.
    aqi_fg.insert(df)
    print("Success! The Feature Store is updated.")
else:
    print("⚠️ No data to upload. Check API responses or Date range.")