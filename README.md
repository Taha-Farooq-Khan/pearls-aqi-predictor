# 🌫️ Multan AQI Predictor (Pearls AQI)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pearls-aqi-predictor.streamlit.app/)
[![Hourly Data Feeder](https://github.com/Taha-Farooq-Khan/pearls-aqi-predictor/actions/workflows/hourly_feature_pipeline.yml/badge.svg)](https://github.com/Taha-Farooq-Khan/pearls-aqi-predictor/actions/workflows/hourly_feature_pipeline.yml)
[![Daily Model Retraining](https://github.com/Taha-Farooq-Khan/pearls-aqi-predictor/actions/workflows/daily_training_pipeline.yml/badge.svg)](https://github.com/Taha-Farooq-Khan/pearls-aqi-predictor/actions/workflows/daily_training_pipeline.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Hopsworks](https://img.shields.io/badge/Feature%20Store-Hopsworks-orange)](https://www.hopsworks.ai/)

**A fully automated End-to-End MLOps pipeline** that predicts PM2.5 air quality levels for Multan, Pakistan. The system runs on a **self-healing schedule**, continuously fetching data, retraining models, and deploying the best performer ("Champion") to a live dashboard.

---

## 🚀 Live Demo
**[Click here to view the Live App](https://pearls-aqi-predictor.streamlit.app/)**
*(Note: If the app is sleeping, click "Wake Up" and wait a moment).*

---

## 🧠 System Architecture

This project is not just a model; it is a **living system**. It consists of three automated pipelines:

### 1. 📡 Hourly Data Pipeline (The "Feeder")
* **Schedule:** Runs every hour (GitHub Actions).
* **Action:** Fetches real-time weather & pollution data from OpenMeteo APIs.
* **Destination:** Pushes data to the **Hopsworks Feature Store**, creating a historical archive of air quality in Multan.

### 2. 🥊 Daily Training Pipeline (The "Gym")
* **Schedule:** Runs every day at 5:00 AM PKT (Midnight UTC).
* **Action:**
    1.  Loads the latest historical data from Hopsworks.
    2.  Trains **4 Competing Models**:
        * **LSTM** (Deep Learning - Long Short-Term Memory)
        * **1D-CNN** (Deep Learning - Convolutional Neural Network)
        * **CatBoost** (Gradient Boosting)
        * **LightGBM** (Gradient Boosting)
    3.  **Evaluates** them using RMSE (Root Mean Squared Error).
    4.  **Promotes** the winner to the **Model Registry** as the new "Champion".

### 3. 📊 Inference Pipeline (The "App")
* **Interface:** Streamlit Dashboard.
* **Action:**
    * Connects to the Model Registry.
    * Downloads the current "Champion" model automatically.
    * Generates a **72-hour forecast** of PM2.5 levels.
    * Displays a **Model Leaderboard** showing current accuracy metrics.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Feature Store:** Hopsworks (Serverless)
* **Orchestration:** GitHub Actions (Cron Jobs)
* **Modeling:** TensorFlow (Keras), Scikit-Learn, CatBoost, LightGBM
* **Frontend:** Streamlit
* **Visualization:** Plotly Interactive Charts

---

## 📂 Project Structure

```text
pearls_aqi_predictor/
├── .github/workflows/       # The "Robots" (Automation scripts)
│   ├── hourly_feature_pipeline.yml
│   └── daily_training_pipeline.yml
├── app/
│   └── streamlit_app.py     # The Dashboard code
├── features/
│   ├── 1_backfill.py        # Initial data loading (One-time)
│   ├── 2_feature_pipeline.py# The hourly script
│   └── 3_training_pipeline.py # The daily training & evaluation script
├── notebooks/               # Experimental / EDA work
│   └── eda_aqi.ipynb
├── requirements.txt         # Project dependencies
└── README.md                # Documentation

```
## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Taha-Farooq-Khan/pearls-aqi-predictor.git](https://github.com/Taha-Farooq-Khan/pearls-aqi-predictor.git)
   cd pearls_aqi_predictor
   ```
2. **Install dependencies::**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up credentials:**
   ```bash
   HOPSWORKS_API_KEY=your_api_key_here
   ```
4. **Run the App**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   
## 🏆 Model Leaderboard logic

The system automatically compares models based on **RMSE** (Root Mean Squared Error) on a hold-out test set (last 10% of data).

| Model | Type | Best For |
| :--- | :--- | :--- |
| **LSTM** | Deep Learning | Capturing long-term time dependencies. |
| **CNN** | Deep Learning | Detecting short-term patterns and spikes. |
| **CatBoost** | Gradient Boosting | Handling tabular weather data efficiently. |
| **LightGBM** | Gradient Boosting | Speed and performance on smaller datasets. |

*The dashboard always uses the model with the lowest RMSE from the latest training run.*

------------------------------------
## 👨‍💻 Author

### **Taha Farooq**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/taha-farooq-khan)
* Data Science & MLOps Enthusiast
* Focus: Building automated AI systems for real-world problems.