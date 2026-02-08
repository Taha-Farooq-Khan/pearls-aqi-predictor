import os
import pandas as pd
import numpy as np
import hopsworks
import joblib
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
from pathlib import Path

# AUTHENTICATION
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv('HOPSWORKS_API_KEY')

print("Connecting to Feature Store...")
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()

# FETCH FULL TRAINING DATA
print("Downloading FULL dataset from Cloud...")
aqi_fg = fs.get_feature_group(name="aqi_data_multan", version=1)
df = aqi_fg.read()

# Sort by time to respect the timeline
df = df.sort_values(by="timestamp")

# PREPROCESSING
print("Preprocessing...")
features = ['pm25_lag_1', 'pm25_lag_6', 'pm25_lag_24', 'pm25_rolling_mean_24h',
            'temp', 'humidity', 'wind_speed', 'rain', 'hour_sin', 'hour_cos', 'humid_temp_interaction']
target = 'pm25'

# --- FIX 1: Split BEFORE Scaling (Prevents Leakage) ---
train_size = int(len(df) * 0.9)
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

# Scale Inputs (X) only
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(df_train[features])
X_test = scaler_X.transform(df_test[features])  # Use the same scaler logic on test!

# --- FIX 2: Do NOT Scale Target (y) ---
# We want the model to predict real PM2.5 values (e.g., 150), not 0.5
y_train = df_train[target].values
y_test = df_test[target].values

# A. Data for ML Models (2D)
X_train_2d, X_test_2d = X_train, X_test

# B. Data for Deep Learning (3D: Rows x 1 Step x Columns)
X_train_3d = X_train_2d.reshape((X_train_2d.shape[0], 1, X_train_2d.shape[1]))
X_test_3d = X_test_2d.reshape((X_test_2d.shape[0], 1, X_test_2d.shape[1]))

print(f"Training Data: {len(X_train_2d)} rows")
print(f"Testing Data:  {len(X_test_2d)} rows")

# TRAIN 4 MODELS
results = {}
model_dir = "aqi_models"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# MODEL 1: LIGHTGBM
print("\n Training LightGBM...")
lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
lgbm.fit(X_train_2d, y_train)
preds_lgbm = lgbm.predict(X_test_2d)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, preds_lgbm))
r2_lgbm = r2_score(y_test, preds_lgbm)
print(f" RMSE: {rmse_lgbm:.4f}")
results["LightGBM"] = {"model": lgbm, "rmse": rmse_lgbm, "r2": r2_lgbm, "type": "sklearn"}

# MODEL 2: CATBOOST
print("\n Training CatBoost...")
cat = CatBoostRegressor(n_estimators=300, learning_rate=0.05, verbose=0, random_state=42)
cat.fit(X_train_2d, y_train)
preds_cat = cat.predict(X_test_2d)
rmse_cat = np.sqrt(mean_squared_error(y_test, preds_cat))
r2_cat = r2_score(y_test, preds_cat)
print(f" RMSE: {rmse_cat:.4f}")
results["CatBoost"] = {"model": cat, "rmse": rmse_cat, "r2": r2_cat, "type": "sklearn"}

# MODEL 3: LSTM
print("\n Training LSTM...")
lstm = Sequential([
    Input(shape=(1, len(features))),
    LSTM(64, activation='relu', return_sequences=False),
    Dense(1)  # Linear output for real values
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_3d, y_train, epochs=50, batch_size=32, verbose=0)  # Increased epochs slightly
preds_lstm = lstm.predict(X_test_3d, verbose=0)
rmse_lstm = np.sqrt(mean_squared_error(y_test, preds_lstm))
r2_lstm = r2_score(y_test, preds_lstm)
print(f" RMSE: {rmse_lstm:.4f}")
results["LSTM"] = {"model": lstm, "rmse": rmse_lstm, "r2": r2_lstm, "type": "keras"}

# MODEL 4: 1D-CNN
print("\n Training 1D-CNN...")
cnn = Sequential([
    Input(shape=(1, len(features))),
    Conv1D(filters=64, kernel_size=1, activation='relu'),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])
cnn.compile(optimizer='adam', loss='mse')
cnn.fit(X_train_3d, y_train, epochs=50, batch_size=32, verbose=0)
preds_cnn = cnn.predict(X_test_3d, verbose=0)
rmse_cnn = np.sqrt(mean_squared_error(y_test, preds_cnn))
r2_cnn = r2_score(y_test, preds_cnn)
print(f" RMSE: {rmse_cnn:.4f}")
results["CNN"] = {"model": cnn, "rmse": rmse_cnn, "r2": r2_cnn, "type": "keras"}

# 5. CHOOSE WINNER
best_model_name = min(results, key=lambda k: results[k]["rmse"])
print(f"\n 🏆 The Winner is: {best_model_name} (RMSE: {results[best_model_name]['rmse']:.4f})")

# 6. UPLOAD TO REGISTRY
mr = project.get_model_registry()

for name, data in results.items():
    print(f" Saving {name}...")

    # Save Locally
    if data["type"] == "sklearn":
        filename = f"{model_dir}/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(data["model"], filename)
    else:
        filename = f"{model_dir}/{name.replace(' ', '_').lower()}.keras"
        data["model"].save(filename)

    # Create Metadata
    is_best = (name == best_model_name)

    # Delete old version if exists (Optional cleanup, but keeps registry clean)
    try:
        old_model = mr.get_model(f"aqi_{name.replace(' ', '_').lower()}_multan", version=None)
        # We don't delete, we just let Hopsworks create a new version
    except:
        pass

    hw_model = mr.python.create_model(
        name=f"aqi_{name.replace(' ', '_').lower()}_multan",
        metrics={"rmse": data["rmse"], "r2": data["r2"], "is_best": 1 if is_best else 0},
        description=f"{name} Model (Real Values)"
    )
    hw_model.save(filename)

print("Success! Models trained and uploaded.")