import pandas as pd
import requests
import hopsworks
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path


# 1. AUTHENTICATION & CONFIGURATION
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv('HOPSWORKS_API_KEY')
if not API_KEY:
    raise ValueError("ERROR! API Key not found! Check your .env/secrets file.")

LAT = 30.1968
LON = 71.4782


# 2. CONNECT TO HOPSWORKS FEATURE STORE
print("Connecting to Feature Store...")
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()

# Connect to the EXISTING Feature Group
aqi_fg = fs.get_feature_group(name="aqi_data_multan", version=1)


# 3. FETCH RECENT DATA (3-Day Window for Lags)
end_date = datetime.now()
start_date = end_date - timedelta(days=3)

print(f"Fetching fresh data ({start_date.date()} to {end_date.date()})...")

# A. Weather (Using Forecast API to avoid Archive API delays)
weather_url = "https://api.open-meteo.com/v1/forecast"
params_w = {
    "latitude": LAT, "longitude": LON,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain",
    "timezone": "auto"
}
response_w = requests.get(weather_url, params=params_w)

if response_w.status_code != 200:
    raise Exception(f"Weather API Error: {response_w.text}")

df_w = pd.DataFrame(response_w.json()['hourly'])

# B. Pollution
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

# 4. DATA PROCESSING & MERGING
# Rename columns to match our expected schema
df_w.rename(columns={
    'time': 'date',
    'temperature_2m': 'temp',
    'relative_humidity_2m': 'humidity',
    'wind_speed_10m': 'wind_speed'
}, inplace=True)

df_a.rename(columns={'time': 'date', 'pm2_5': 'pm25'}, inplace=True)

# Merge datasets on the date column
df_w['date'] = pd.to_datetime(df_w['date'])
df_a['date'] = pd.to_datetime(df_a['date'])
df = pd.merge(df_w, df_a, on='date')


# 5. FEATURE ENGINEERING
# Cumulative Lags
df['pm25_lag_1'] = df['pm25'].shift(1)
df['pm25_lag_6'] = df['pm25'].shift(6)
df['pm25_lag_24'] = df['pm25'].shift(24)

# Rolling Trends
df['pm25_rolling_mean_24h'] = df['pm25'].rolling(window=24).mean()
df['pm25_rolling_std_24h'] = df['pm25'].rolling(window=24).std()

# Cyclical Time
df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)

# Interactions
df['humid_temp_interaction'] = df['humidity'] * df['temp']

# Hopsworks BigInt Timestamp Requirement
df['timestamp'] = (df['date'].values.astype("int64") // 10 ** 6).astype("int64")


# 6. DATA CLEANING & UPLOAD
# Filter out future dates provided by the daily forecast
current_time = datetime.now()
df = df[df['date'] <= current_time]

# Drop NaNs (Drops the first 24h of our 3-day window due to lag_24)
df = df.dropna()

print(f"New Data Processed: {len(df)} rows.")

if len(df) > 0:
    print("Pushing to Cloud...")

    # insert() updates existing rows and inserts new ones using the 'timestamp' primary key.
    # wait=False prevents GitHub Action timeout errors by executing asynchronously.
    aqi_fg.insert(df, wait=False)

    print("Success! The Feature Store is updated (Materialization running in background).")
else:
    print("No data to upload. Check API responses or Date range.")