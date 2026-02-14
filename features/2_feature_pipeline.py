"""
AQI Feature Pipeline - Data Ingestion & Quality Control
========================================================
Fetches weather and pollution data, performs basic validation,
and stores RAW data in Hopsworks Feature Store.

Runs: Hourly via GitHub Actions
"""

import pandas as pd
import requests
import hopsworks
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AQIDataIngestion:
    """Handles data fetching, validation, and upload to Hopsworks"""

    def __init__(self):
        """Initialize configuration and connections"""
        # Load environment variables
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)

        self.api_key = os.getenv('HOPSWORKS_API_KEY')
        if not self.api_key:
            raise ValueError("ERROR! HOPSWORKS_API_KEY not found in .env file")

        # Location: Multan, Pakistan
        self.lat = 30.1968
        self.lon = 71.4782

        # Data quality thresholds
        self.pm25_min = 0
        self.pm25_max = 500
        self.temp_min = -20
        self.temp_max = 55
        self.humidity_min = 0
        self.humidity_max = 100
        self.max_pm25_jump = 150  # μg/m³ per hour

        # Initialize Hopsworks connection
        logger.info("Connecting to Hopsworks Feature Store...")
        self.project = hopsworks.login(api_key_value=self.api_key)
        self.fs = self.project.get_feature_store()

        # Feature group name (matches your Streamlit app)
        self.feature_group_name = "aqi_data_multan"
        self.feature_group_version = 1

    def fetch_weather_data(self, start_date, end_date):
        """Fetch weather data from Open-Meteo API"""
        logger.info(f"Fetching weather data from {start_date.date()} to {end_date.date()}...")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain",
            "timezone": "UTC"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data['hourly'])

            # Rename columns
            df.rename(columns={
                'time': 'date',
                'temperature_2m': 'temp',
                'relative_humidity_2m': 'humidity',
                'wind_speed_10m': 'wind_speed'
            }, inplace=True)

            logger.info(f"Weather data fetched: {len(df)} rows")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise

    def fetch_pollution_data(self, start_date, end_date):
        """Fetch air quality data from Open-Meteo Air Quality API"""
        logger.info(f"Fetching pollution data from {start_date.date()} to {end_date.date()}...")

        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "pm10,pm2_5",
            "timezone": "UTC"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data['hourly'])

            # Rename columns
            df.rename(columns={
                'time': 'date',
                'pm2_5': 'pm25'
            }, inplace=True)

            logger.info(f"Pollution data fetched: {len(df)} rows")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Pollution API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise

    def validate_data_quality(self, df):
        """
        Validate data quality and return issues found

        Returns:
            tuple: (is_valid: bool, issues: list)
        """
        issues = []

        # Check for required columns
        required_cols = ['date', 'temp', 'humidity', 'wind_speed', 'rain', 'pm25', 'pm10']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return False, issues

        # Check PM2.5 range
        pm25_out_of_range = (df['pm25'] < self.pm25_min) | (df['pm25'] > self.pm25_max)
        if pm25_out_of_range.any():
            count = pm25_out_of_range.sum()
            min_val, max_val = df['pm25'].min(), df['pm25'].max()
            issues.append(f"PM2.5 out of range [{self.pm25_min}, {self.pm25_max}]: "
                          f"{count} rows (range: {min_val:.1f} - {max_val:.1f})")

        # Check temperature range
        temp_out_of_range = (df['temp'] < self.temp_min) | (df['temp'] > self.temp_max)
        if temp_out_of_range.any():
            count = temp_out_of_range.sum()
            issues.append(f"Temperature out of range [{self.temp_min}, {self.temp_max}]: "
                          f"{count} rows")

        # Check humidity range
        humidity_out_of_range = (df['humidity'] < self.humidity_min) | (df['humidity'] > self.humidity_max)
        if humidity_out_of_range.any():
            count = humidity_out_of_range.sum()
            issues.append(f"Humidity out of range [{self.humidity_min}, {self.humidity_max}]: "
                          f"{count} rows")

        # Check for suspicious PM2.5 jumps
        df_sorted = df.sort_values('date')
        pm25_diff = df_sorted['pm25'].diff().abs()
        large_jumps = pm25_diff > self.max_pm25_jump
        if large_jumps.any():
            count = large_jumps.sum()
            max_jump = pm25_diff.max()
            issues.append(f"Suspicious PM2.5 jumps (>{self.max_pm25_jump} μg/m³/hr): "
                          f"{count} occurrences (max: {max_jump:.1f})")

        # Check missing data percentage
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 5]
        if not high_missing.empty:
            issues.append(f"High missing data (>5%): {high_missing.to_dict()}")

        # Check for duplicate timestamps
        duplicates = df['date'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate timestamps found: {duplicates} rows")

        # Determine if data is valid (only critical issues fail validation)
        critical_issues = [
            issue for issue in issues
            if "Missing required columns" in issue or "Duplicate timestamps" in issue
        ]

        is_valid = len(critical_issues) == 0

        return is_valid, issues

    def get_existing_data(self):
        """Fetch existing data from feature store"""
        try:
            fg = self.fs.get_feature_group(
                name=self.feature_group_name,
                version=self.feature_group_version
            )
            df = fg.read()
            logger.info(f"Existing data in feature store: {len(df)} rows")
            return df, fg
        except Exception as e:
            logger.info(f"Feature group doesn't exist yet or is empty: {e}")
            return pd.DataFrame(), None

    def remove_duplicates(self, new_df, existing_df):
        """Remove timestamps that already exist in the feature store"""
        if existing_df.empty:
            logger.info("No existing data - all new data will be inserted")
            return new_df

        # Convert to datetime and normalize to hour
        new_df['date'] = pd.to_datetime(new_df['date'])
        existing_df['date'] = pd.to_datetime(existing_df['date'])

        existing_dates = set(existing_df['date'].dt.floor('H'))
        before_count = len(new_df)

        new_df = new_df[~new_df['date'].dt.floor('H').isin(existing_dates)]
        after_count = len(new_df)

        removed = before_count - after_count
        if removed > 0:
            logger.info(f"Removed {removed} duplicate timestamps")
        else:
            logger.info("No duplicates found")

        return new_df

    def prepare_for_upload(self, df):
        """Prepare dataframe for Hopsworks upload"""
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Filter out future dates (forecast API sometimes returns future data)
        current_time = datetime.now(timezone.utc)
        df = df[df['date'] <= current_time]
        logger.info(f"After filtering future dates: {len(df)} rows")

        # Create BigInt timestamp for Hopsworks (milliseconds since epoch)
        df['timestamp'] = (df['date'].values.astype("int64") // 10 ** 6).astype("int64")

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Drop any remaining NaN values
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)

        if rows_before != rows_after:
            logger.warning(f"Dropped {rows_before - rows_after} rows with NaN values")

        return df

    def create_feature_group(self):
        """Create feature group if it doesn't exist"""
        try:
            fg = self.fs.get_feature_group(
                name=self.feature_group_name,
                version=self.feature_group_version
            )
            logger.info("Feature group already exists")
            return fg
        except:
            logger.info("Creating new feature group...")
            fg = self.fs.create_feature_group(
                name=self.feature_group_name,
                version=self.feature_group_version,
                description="Raw weather and air quality data for Multan, Pakistan",
                primary_key=["timestamp"],
                event_time="timestamp",
                online_enabled=False
            )
            logger.info("Feature group created successfully")
            return fg

    def run(self, days_back=3):
        """
        Main execution method

        Args:
            days_back (int): Number of days to fetch (default: 3)
        """
        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)

            logger.info("=" * 60)
            logger.info("AQI DATA INGESTION PIPELINE")
            logger.info("=" * 60)

            # Fetch data from APIs
            df_weather = self.fetch_weather_data(start_date, end_date)
            df_pollution = self.fetch_pollution_data(start_date, end_date)

            # Merge datasets
            df_weather['date'] = pd.to_datetime(df_weather['date'])
            df_pollution['date'] = pd.to_datetime(df_pollution['date'])
            df = pd.merge(df_weather, df_pollution, on='date', how='inner')
            logger.info(f"Merged data: {len(df)} rows")

            # Validate data quality
            logger.info("Validating data quality...")
            is_valid, issues = self.validate_data_quality(df)

            if issues:
                logger.warning("Data quality issues detected:")
                for issue in issues:
                    logger.warning(f"  ⚠️  {issue}")
            else:
                logger.info("✓ Data quality check passed - no issues found")

            if not is_valid:
                logger.error("Critical data quality issues found. Aborting upload.")
                return False

            # Get existing data and remove duplicates
            existing_df, _ = self.get_existing_data()
            df = self.remove_duplicates(df, existing_df)

            if len(df) == 0:
                logger.info("No new data to upload. All timestamps already exist.")
                return True

            # Prepare for upload
            df = self.prepare_for_upload(df)
            logger.info(f"Final data ready for upload: {len(df)} rows")

            # Create or get feature group
            fg = self.create_feature_group()

            # Upload to Hopsworks
            logger.info("Uploading to Hopsworks Feature Store...")
            try:
                fg.insert(df, wait=False)
                logger.info("✓ SUCCESS! Data uploaded to Feature Store")
                logger.info(f"  - Rows uploaded: {len(df)}")
                logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
                return True

            except Exception as e:
                error_msg = str(e)
                # Handle known connection issues that occur after successful upload
                if "RemoteDisconnected" in error_msg or "Connection aborted" in error_msg:
                    logger.warning(f"⚠️  Connection lost after upload: {error_msg}")
                    logger.info("Assuming data upload was successful.")
                    logger.info("Check Hopsworks UI to confirm materialization job status.")
                    return True
                else:
                    logger.error(f"Upload failed: {e}")
                    raise

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def main():
    """Entry point for the feature pipeline"""
    pipeline = AQIDataIngestion()
    success = pipeline.run(days_back=3)

    if success:
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 60)
        exit(1)


if __name__ == "__main__":
    main()