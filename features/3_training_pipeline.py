import os
import pandas as pd
import numpy as np
import hopsworks
import joblib
import shap  # ADDED: For Explainable AI
import matplotlib.pyplot as plt  # ADDED: For plotting SHAP
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # ADDED: MAE
from dotenv import load_dotenv
from pathlib import Path


# 1. AUTHENTICATION & CONNECTION
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv('HOPSWORKS_API_KEY')

print("Connecting to Feature Store...")
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()


# 2. FETCH & PREPROCESS DATA

print("Downloading FULL dataset from Cloud...")
aqi_fg = fs.get_feature_group(name="aqi_data_multan", version=1)
df = aqi_fg.read()

# Sort by time to respect the timeline
df = df.sort_values(by="timestamp")

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
y_train = df_train[target].values
y_test = df_test[target].values

# A. Data for ML Models (2D)
X_train_2d, X_test_2d = X_train, X_test

# B. Data for Deep Learning (3D: Rows x 1 Step x Columns)
X_train_3d = X_train_2d.reshape((X_train_2d.shape[0], 1, X_train_2d.shape[1]))
X_test_3d = X_test_2d.reshape((X_test_2d.shape[0], 1, X_test_2d.shape[1]))

print(f"Training Data: {len(X_train_2d)} rows")
print(f"Testing Data:  {len(X_test_2d)} rows")


# 3. TRAIN MODELS & EVALUATE (Now with MAE)
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
mae_lgbm = mean_absolute_error(y_test, preds_lgbm)
r2_lgbm = r2_score(y_test, preds_lgbm)
print(f" RMSE: {rmse_lgbm:.4f} | MAE: {mae_lgbm:.4f} | R²: {r2_lgbm:.4f}")
results["LightGBM"] = {"model": lgbm, "rmse": rmse_lgbm, "mae": mae_lgbm, "r2": r2_lgbm, "type": "sklearn"}

# MODEL 2: CATBOOST
print("\n Training CatBoost...")
cat = CatBoostRegressor(n_estimators=300, learning_rate=0.05, verbose=0, random_state=42)
cat.fit(X_train_2d, y_train)
preds_cat = cat.predict(X_test_2d)
rmse_cat = np.sqrt(mean_squared_error(y_test, preds_cat))
mae_cat = mean_absolute_error(y_test, preds_cat)
r2_cat = r2_score(y_test, preds_cat)
print(f" RMSE: {rmse_cat:.4f} | MAE: {mae_cat:.4f} | R²: {r2_cat:.4f}")
results["CatBoost"] = {"model": cat, "rmse": rmse_cat, "mae": mae_cat, "r2": r2_cat, "type": "sklearn"}

# MODEL 3: LSTM
print("\n Training LSTM...")
lstm = Sequential([
    Input(shape=(1, len(features))),
    LSTM(64, activation='relu', return_sequences=False),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_3d, y_train, epochs=50, batch_size=32, verbose=0)
preds_lstm = lstm.predict(X_test_3d, verbose=0)
rmse_lstm = np.sqrt(mean_squared_error(y_test, preds_lstm))
mae_lstm = mean_absolute_error(y_test, preds_lstm)
r2_lstm = r2_score(y_test, preds_lstm)
print(f" RMSE: {rmse_lstm:.4f} | MAE: {mae_lstm:.4f} | R²: {r2_lstm:.4f}")
results["LSTM"] = {"model": lstm, "rmse": rmse_lstm, "mae": mae_lstm, "r2": r2_lstm, "type": "keras"}

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
mae_cnn = mean_absolute_error(y_test, preds_cnn)
r2_cnn = r2_score(y_test, preds_cnn)
print(f" RMSE: {rmse_cnn:.4f} | MAE: {mae_cnn:.4f} | R²: {r2_cnn:.4f}")
results["CNN"] = {"model": cnn, "rmse": rmse_cnn, "mae": mae_cnn, "r2": r2_cnn, "type": "keras"}


# 4. CHOOSE WINNER & GENERATE SHAP
best_model_name = min(results, key=lambda k: results[k]["rmse"])
print(f"\nThe Winner is: {best_model_name} (RMSE: {results[best_model_name]['rmse']:.4f})")

# --- SHAP EXPLAINABILITY (Only for Tree models as requested) ---
print("\nGenerating SHAP Explainer Plot for the best Tree-based model...")
# Find the best Tree model (LightGBM or CatBoost) for SHAP
tree_models = {k: v for k, v in results.items() if v["type"] == "sklearn"}
best_tree_name = min(tree_models, key=lambda k: tree_models[k]["rmse"])
best_tree_model = tree_models[best_tree_name]["model"]

# Convert scaled X_test back to DataFrame so SHAP can label the features correctly
X_test_df = pd.DataFrame(X_test_2d, columns=features)
X_test_sample = X_test_df.sample(n=min(1000, len(X_test_df)), random_state=42)

explainer = shap.TreeExplainer(best_tree_model)
shap_values = explainer.shap_values(X_test_sample)

plt.figure(figsize=(10, 6))
plt.title(f"SHAP Feature Importance: {best_tree_name}", fontsize=14)
shap.summary_plot(shap_values, X_test_sample, show=False)
shap_filename = f"shap_summary_{best_tree_name.lower()}.png"
plt.savefig(shap_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"SHAP plot saved locally as '{shap_filename}'")

# 5. UPLOAD TO HOPSWORKS REGISTRY
mr = project.get_model_registry()

for name, data in results.items():
    print(f"Saving {name} to Model Registry...")

    # Save Locally
    if data["type"] == "sklearn":
        filename = f"{model_dir}/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(data["model"], filename)
    else:
        filename = f"{model_dir}/{name.replace(' ', '_').lower()}.keras"
        data["model"].save(filename)

    is_best = (name == best_model_name)

    # ADDED MAE to Hopsworks Metadata Tracking
    hw_model = mr.python.create_model(
        name=f"aqi_{name.replace(' ', '_').lower()}_multan",
        metrics={
            "rmse": data["rmse"],
            "mae": data["mae"],  # <-- MAE logged to Hopsworks!
            "r2": data["r2"],
            "is_best": 1 if is_best else 0
        },
        description=f"{name} Model (Real Values)"
    )
    hw_model.save(filename)

print("Success! Models trained, evaluated, and uploaded.")