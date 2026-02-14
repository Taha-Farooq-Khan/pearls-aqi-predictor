"""
AQI Training Pipeline - Model Training & Evaluation
====================================================
Fetches raw data, performs feature engineering on the complete dataset,
splits into train/val/test, trains multiple models, and uploads the best
model to Hopsworks Model Registry.

Runs: Daily via GitHub Actions
"""

import os
import pandas as pd
import numpy as np
import hopsworks
import joblib
import shap
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for GitHub Actions
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dotenv import load_dotenv
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AQITrainingPipeline:
    """Complete training pipeline for AQI prediction models"""

    def __init__(self):
        """Initialize configuration and connections"""
        # Load environment variables
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)

        self.api_key = os.getenv('HOPSWORKS_API_KEY')
        if not self.api_key:
            raise ValueError("ERROR! HOPSWORKS_API_KEY not found in .env file")

        # Feature configuration (matches your Streamlit app)
        self.feature_names = [
            'pm25_lag_1', 'pm25_lag_6', 'pm25_lag_24',
            'pm25_rolling_mean_24h',
            'temp', 'humidity', 'wind_speed', 'rain',
            'hour_sin', 'hour_cos',
            'humid_temp_interaction'
        ]
        self.target = 'pm25'

        # Model directory
        self.model_dir = "aqi_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Results storage
        self.results = {}
        self.scaler_X = None

        # Connect to Hopsworks
        logger.info("Connecting to Hopsworks Feature Store...")
        self.project = hopsworks.login(api_key_value=self.api_key)
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()

    def fetch_data(self):
        """Fetch raw data from feature store"""
        logger.info("Fetching data from Feature Store...")

        try:
            fg = self.fs.get_feature_group(name="aqi_data_multan", version=1)
            df = fg.read()

            # Sort by timestamp
            df = df.sort_values(by="timestamp")
            df['date'] = pd.to_datetime(df['date'])

            logger.info(f"Data fetched: {len(df)} rows")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

    def engineer_features(self, df):
        """
        Create all features on the complete dataset
        This prevents leakage - features are created before splitting
        """
        logger.info("Engineering features...")

        # Ensure sorted by time
        df = df.sort_values('date').reset_index(drop=True)

        # 1. LAG FEATURES (using shift - no future data leakage)
        df['pm25_lag_1'] = df['pm25'].shift(1)
        df['pm25_lag_6'] = df['pm25'].shift(6)
        df['pm25_lag_24'] = df['pm25'].shift(24)

        # 2. ROLLING STATISTICS (shifted to prevent leakage)
        # CRITICAL: Use shift(1) so the rolling mean only includes past data
        df['pm25_rolling_mean_24h'] = df['pm25'].shift(1).rolling(window=24, min_periods=1).mean()

        # 3. CYCLICAL TIME FEATURES
        df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)

        # 4. INTERACTION FEATURES
        df['humid_temp_interaction'] = df['humidity'] * df['temp']

        # 5. Remove rows with NaN (due to lag/rolling calculations)
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        logger.info(f"Dropped {rows_before - rows_after} rows with NaN (expected due to lags)")

        logger.info(f"Feature engineering complete: {len(df)} rows, {len(self.feature_names)} features")

        return df

    def split_data(self, df, train_ratio=0.70, val_ratio=0.15):
        """
        Split data into train/val/test with no leakage

        Time series split:
        - Train: 70%
        - Validation: 15% (for model selection & hyperparameter tuning)
        - Test: 15% (for final evaluation only)
        """
        logger.info("Splitting data into train/val/test...")

        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * (train_ratio + val_ratio))

        df_train = df.iloc[:train_size].copy()
        df_val = df.iloc[train_size:val_size].copy()
        df_test = df.iloc[val_size:].copy()

        logger.info(f"Train set: {len(df_train)} rows ({df_train['date'].min()} to {df_train['date'].max()})")
        logger.info(f"Val set:   {len(df_val)} rows ({df_val['date'].min()} to {df_val['date'].max()})")
        logger.info(f"Test set:  {len(df_test)} rows ({df_test['date'].min()} to {df_test['date'].max()})")

        return df_train, df_val, df_test

    def scale_features(self, df_train, df_val, df_test):
        """
        Scale features using MinMaxScaler
        Fit on train, transform on val/test
        """
        logger.info("Scaling features...")

        # Fit scaler on training data only
        self.scaler_X = MinMaxScaler()
        X_train = self.scaler_X.fit_transform(df_train[self.feature_names])

        # Transform validation and test data using the same scaler
        X_val = self.scaler_X.transform(df_val[self.feature_names])
        X_test = self.scaler_X.transform(df_test[self.feature_names])

        # Target values (NOT scaled)
        y_train = df_train[self.target].values
        y_val = df_val[self.target].values
        y_test = df_test[self.target].values

        logger.info("Scaling complete")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def evaluate_model(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        logger.info("Training LightGBM...")

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )

        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Metrics
        train_metrics = self.evaluate_model(y_train, train_pred)
        val_metrics = self.evaluate_model(y_val, val_pred)

        logger.info(
            f"  Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        logger.info(
            f"  Val   - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")

        return model, train_metrics, val_metrics

    def train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model"""
        logger.info("Training CatBoost...")

        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=7,
            random_seed=42,
            verbose=0
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Metrics
        train_metrics = self.evaluate_model(y_train, train_pred)
        val_metrics = self.evaluate_model(y_val, val_pred)

        logger.info(
            f"  Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        logger.info(
            f"  Val   - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")

        return model, train_metrics, val_metrics

    def train_lstm(self, X_train, y_train, X_val, y_val):
        """Train LSTM model"""
        logger.info("Training LSTM...")

        # Reshape for LSTM: (samples, timesteps=1, features)
        X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_3d = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        model = Sequential([
            Input(shape=(1, len(self.feature_names))),
            LSTM(64, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        # Train
        model.fit(
            X_train_3d, y_train,
            validation_data=(X_val_3d, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # Predictions
        train_pred = model.predict(X_train_3d, verbose=0).flatten()
        val_pred = model.predict(X_val_3d, verbose=0).flatten()

        # Metrics
        train_metrics = self.evaluate_model(y_train, train_pred)
        val_metrics = self.evaluate_model(y_val, val_pred)

        logger.info(
            f"  Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        logger.info(
            f"  Val   - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")

        return model, train_metrics, val_metrics

    def train_cnn(self, X_train, y_train, X_val, y_val):
        """Train 1D-CNN model"""
        logger.info("Training CNN...")

        # Reshape for CNN: (samples, timesteps=1, features)
        X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_3d = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        model = Sequential([
            Input(shape=(1, len(self.feature_names))),
            Conv1D(filters=64, kernel_size=1, activation='relu'),
            MaxPooling1D(pool_size=1),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train
        model.fit(
            X_train_3d, y_train,
            validation_data=(X_val_3d, y_val),
            epochs=50,
            batch_size=32,
            verbose=0
        )

        # Predictions
        train_pred = model.predict(X_train_3d, verbose=0).flatten()
        val_pred = model.predict(X_val_3d, verbose=0).flatten()

        # Metrics
        train_metrics = self.evaluate_model(y_train, train_pred)
        val_metrics = self.evaluate_model(y_val, val_pred)

        logger.info(
            f"  Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        logger.info(
            f"  Val   - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")

        return model, train_metrics, val_metrics

    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all models and store results"""
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)

        # Models to train (matches your Streamlit app expectations)
        models_to_train = [
            ('lightgbm', self.train_lightgbm),
            ('catboost', self.train_catboost),
            ('lstm', self.train_lstm),
            ('cnn', self.train_cnn)
        ]

        for name, train_func in models_to_train:
            try:
                model, train_metrics, val_metrics = train_func(X_train, y_train, X_val, y_val)

                self.results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'model_type': 'tree' if name in ['lightgbm', 'catboost'] else 'deep_learning'
                }

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info("\n" + "=" * 60)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("=" * 60)

    def select_best_model(self):
        """Select best model based on validation RMSE"""
        logger.info("Selecting best model based on validation RMSE...")

        best_model_name = min(
            self.results.keys(),
            key=lambda k: self.results[k]['val_metrics']['rmse']
        )

        logger.info(f"\n🏆 WINNER: {best_model_name.upper()}")
        logger.info(f"  Validation RMSE: {self.results[best_model_name]['val_metrics']['rmse']:.4f}")
        logger.info(f"  Validation MAE:  {self.results[best_model_name]['val_metrics']['mae']:.4f}")
        logger.info(f"  Validation R²:   {self.results[best_model_name]['val_metrics']['r2']:.4f}")

        return best_model_name

    def final_test_evaluation(self, best_model_name, X_test, y_test):
        """
        Final evaluation on test set - DONE ONLY ONCE
        This is the true performance estimate
        """
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("=" * 60)

        model = self.results[best_model_name]['model']

        # Reshape for LSTM/CNN if needed
        if best_model_name in ['lstm', 'cnn']:
            X_test_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            test_pred = model.predict(X_test_input, verbose=0).flatten()
        else:
            test_pred = model.predict(X_test)

        test_metrics = self.evaluate_model(y_test, test_pred)

        logger.info(f"Model: {best_model_name}")
        logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  Test MAE:  {test_metrics['mae']:.4f}")
        logger.info(f"  Test R²:   {test_metrics['r2']:.4f}")

        # Store test metrics
        self.results[best_model_name]['test_metrics'] = test_metrics

        return test_metrics

    def generate_shap_explanations(self, best_model_name, X_val):
        """
        Generate SHAP explanations on VALIDATION set
        (Never use test set for explanations)
        """
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING SHAP EXPLANATIONS")
        logger.info("=" * 60)

        # Only generate SHAP for tree-based models
        if self.results[best_model_name]['model_type'] != 'tree':
            logger.info(f"SHAP not available for {best_model_name} (not a tree model)")
            return None

        model = self.results[best_model_name]['model']

        # Sample from validation set
        sample_size = min(500, len(X_val))
        sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
        X_val_sample = X_val[sample_indices]

        # Create DataFrame for feature names
        X_val_df = pd.DataFrame(X_val_sample, columns=self.feature_names)

        logger.info(f"Computing SHAP values for {len(X_val_sample)} validation samples...")

        try:
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_df)

            # Generate summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_val_df, show=False)
            plt.title(f"SHAP Feature Importance: {best_model_name.upper()}", fontsize=14, pad=20)
            plt.tight_layout()
            summary_path = f"{self.model_dir}/shap_summary_{best_model_name}.png"
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  ✓ SHAP plot saved: {summary_path}")

        except Exception as e:
            logger.warning(f"SHAP generation failed: {e}")

        return True

    def save_model_locally(self, model_name, model):
        """Save model to disk"""
        model_type = self.results[model_name]['model_type']

        if model_type == 'tree':
            filename = f"{self.model_dir}/{model_name}.pkl"
            joblib.dump(model, filename)
        else:  # deep_learning
            filename = f"{self.model_dir}/{model_name}.keras"
            model.save(filename)

        logger.info(f"  ✓ Model saved: {filename}")

        return filename

    def upload_to_model_registry(self, best_model_name):
        """Upload all models to Hopsworks Model Registry"""
        logger.info("\n" + "=" * 60)
        logger.info("UPLOADING TO MODEL REGISTRY")
        logger.info("=" * 60)

        for model_name, data in self.results.items():
            try:
                logger.info(f"\nUploading {model_name}...")

                # Save locally
                model_file = self.save_model_locally(model_name, data['model'])

                is_best = (model_name == best_model_name)

                # Prepare metrics (matching your Streamlit app expectations)
                metrics = {
                    'rmse': data['val_metrics']['rmse'],  # Use val metrics for comparison
                    'mae': data['val_metrics']['mae'],
                    'r2': data['val_metrics']['r2'],
                    'is_best': 1 if is_best else 0
                }

                # Add test metrics for best model
                if is_best and 'test_metrics' in data:
                    metrics['test_rmse'] = data['test_metrics']['rmse']
                    metrics['test_mae'] = data['test_metrics']['mae']
                    metrics['test_r2'] = data['test_metrics']['r2']

                # Create model in registry
                hw_model = self.mr.python.create_model(
                    name=f"aqi_{model_name}_multan",
                    metrics=metrics,
                    description=f"{model_name.upper()} model for PM2.5 prediction in Multan, Pakistan. "
                                f"{'✓ BEST MODEL' if is_best else 'Alternative model'}"
                )

                # Upload model
                hw_model.save(model_file)

                logger.info(f"  ✓ {model_name} uploaded successfully")

            except Exception as e:
                logger.error(f"  ✗ Failed to upload {model_name}: {e}")

    def run(self):
        """Execute the complete training pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("AQI TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            # 1. Fetch data
            df = self.fetch_data()

            # 2. Engineer features on FULL dataset
            df = self.engineer_features(df)

            # 3. Split data (no leakage - features already created)
            df_train, df_val, df_test = self.split_data(df)

            # 4. Scale features
            X_train, X_val, X_test, y_train, y_val, y_test = self.scale_features(
                df_train, df_val, df_test
            )

            # 5. Train all models
            self.train_all_models(X_train, y_train, X_val, y_val)

            # 6. Select best model (based on validation set)
            best_model_name = self.select_best_model()

            # 7. Final evaluation on test set (ONCE only)
            test_metrics = self.final_test_evaluation(best_model_name, X_test, y_test)

            # 8. Generate SHAP explanations (on validation set)
            self.generate_shap_explanations(best_model_name, X_val)

            # 9. Upload to Model Registry
            self.upload_to_model_registry(best_model_name)

            logger.info("\n" + "=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"\n🎉 Best Model: {best_model_name.upper()}")
            logger.info(f"   Test RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"   Test MAE:  {test_metrics['mae']:.4f}")
            logger.info(f"   Test R²:   {test_metrics['r2']:.4f}")

            return True

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def main():
    """Entry point for the training pipeline"""
    pipeline = AQITrainingPipeline()
    success = pipeline.run()

    if not success:
        logger.error("=" * 60)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 60)
        exit(1)


if __name__ == "__main__":
    main()