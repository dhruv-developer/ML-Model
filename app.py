from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
import joblib
import logging
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

# Initialize FastAPI app and logger
app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths and constants
DATA_FILE = "real_time_data.csv"
MODEL_DIR = "product_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load historical data
if not os.path.exists(DATA_FILE):
    data = pd.DataFrame(columns=["date", "product_id", "item_name", "quantity", "stock_level", "department"])
    data.to_csv(DATA_FILE, index=False)
else:
    data = pd.read_csv(DATA_FILE, parse_dates=["date"])

# Request schema for data posting
class DataEntry(BaseModel):
    date: str
    product_id: str
    item_name: str
    quantity: int
    stock_level: int
    department: str

# Train SARIMA model
def train_sarima(y):
    try:
        model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        return results
    except Exception as e:
        logging.error(f"Error training SARIMA: {e}")
        return None

# Train Random Forest model
def train_random_forest(X, y):
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        logging.error(f"Error training Random Forest: {e}")
        return None

# Train models for a specific product
def train_models_for_product(product_id):
    product_data = data[data['product_id'] == product_id].sort_values(by="date")
    y = product_data["quantity"].to_numpy()

    if len(y) < 12:
        logging.warning(f"Insufficient data to train models for product {product_id}.")
        return

    # Train SARIMA
    sarima_model = train_sarima(y)
    if sarima_model:
        sarima_path = os.path.join(MODEL_DIR, f"{product_id}_sarima.pkl")
        joblib.dump(sarima_model, sarima_path)
        logging.info(f"Trained and saved SARIMA model for product {product_id}.")

    # Train Random Forest
    X = np.arange(len(y)).reshape(-1, 1)
    rf_model = train_random_forest(X, y)
    if rf_model:
        rf_path = os.path.join(MODEL_DIR, f"{product_id}_rf.pkl")
        joblib.dump(rf_model, rf_path)
        logging.info(f"Trained and saved Random Forest model for product {product_id}.")

# POST endpoint to accept real-time data
@app.post("/post-data/")
def post_data(entries: List[DataEntry]):
    global data

    new_data = pd.DataFrame([entry.dict() for entry in entries])
    new_data['date'] = pd.to_datetime(new_data['date'])

    data = pd.concat([data, new_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
    data.to_csv(DATA_FILE, index=False)

    affected_products = new_data["product_id"].unique()
    for product_id in affected_products:
        train_models_for_product(product_id)

    return {"message": "Data added and models updated successfully."}

# Predict for 15-day intervals
def predict_for_15_day_intervals(product_id, days):
    predictions = []
    forecast_dates = pd.date_range(start=datetime.now().date(), periods=days, freq='15D')  # 15-day intervals

    product_data = data[data['product_id'] == product_id]
    if product_data.empty:
        logging.warning(f"No data available for product {product_id}.")
        return []

    predictions_dict = {"sarima": [], "rf": []}

    try:
        # SARIMA Model Prediction
        sarima_path = os.path.join(MODEL_DIR, f"{product_id}_sarima.pkl")
        if os.path.exists(sarima_path):
            sarima_model = joblib.load(sarima_path)
            sarima_forecast = sarima_model.get_forecast(steps=days).predicted_mean
            # We sum the forecast for each block of 15 days to get the total quantity for each interval
            predictions_dict["sarima"] = [np.sum(sarima_forecast[i:i+15]) for i in range(0, len(sarima_forecast), 15)]

        # Random Forest Model Prediction
        rf_path = os.path.join(MODEL_DIR, f"{product_id}_rf.pkl")
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            X_future = np.arange(len(product_data), len(product_data) + days).reshape(-1, 1)
            rf_forecast = rf_model.predict(X_future)
            # Similarly, we sum the forecast for each 15-day block
            predictions_dict["rf"] = [np.sum(rf_forecast[i:i+15]) for i in range(0, len(rf_forecast), 15)]

        # Combine forecasts
        forecasts = [predictions_dict[key] for key in predictions_dict if len(predictions_dict[key]) > 0]
        final_forecast = np.mean(forecasts, axis=0) if len(forecasts) > 0 else np.array([])

        for i, quantity in enumerate(final_forecast):
            predictions.append({
                "order_date": forecast_dates[i].date(),
                "item_name": product_data.iloc[0]["item_name"],  # Assuming item_name is the same for the product
                "department": product_data.iloc[0]["department"],  # Assuming department is the same for the product
                "quantity": round(quantity, 2)
            })
    except Exception as e:
        logging.error(f"Error in prediction for product {product_id}: {e}")

    return predictions

# Predict for 1-day intervals (daily)
def predict_for_daily_intervals(product_id):
    predictions = []
    forecast_dates = pd.date_range(start=datetime.now().date(), periods=1, freq='D')  # 1-day interval

    product_data = data[data['product_id'] == product_id]
    if product_data.empty:
        logging.warning(f"No data available for product {product_id}.")
        return []

    predictions_dict = {"sarima": [], "rf": []}

    try:
        # SARIMA Model Prediction
        sarima_path = os.path.join(MODEL_DIR, f"{product_id}_sarima.pkl")
        if os.path.exists(sarima_path):
            sarima_model = joblib.load(sarima_path)
            sarima_forecast = sarima_model.get_forecast(steps=1).predicted_mean
            predictions_dict["sarima"] = [sarima_forecast[0]]

        # Random Forest Model Prediction
        rf_path = os.path.join(MODEL_DIR, f"{product_id}_rf.pkl")
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            X_future = np.arange(len(product_data), len(product_data) + 1).reshape(-1, 1)
            rf_forecast = rf_model.predict(X_future)
            predictions_dict["rf"] = [rf_forecast[0]]

        # Combine forecasts
        forecasts = [predictions_dict[key] for key in predictions_dict if len(predictions_dict[key]) > 0]
        final_forecast = np.mean(forecasts, axis=0) if len(forecasts) > 0 else np.array([])

        for i, quantity in enumerate(final_forecast):
            predictions.append({
                "order_date": forecast_dates[i].date(),
                "item_name": product_data.iloc[0]["item_name"],  # Assuming item_name is the same for the product
                "department": product_data.iloc[0]["department"],  # Assuming department is the same for the product
                "quantity": round(quantity, 2)
            })
    except Exception as e:
        logging.error(f"Error in prediction for product {product_id}: {e}")

    return predictions

# GET endpoint to make predictions for a specific number of days (in 15-day intervals)
@app.get("/predict/")
def predict_for_days_endpoint(days: int = Query(..., ge=1, le=365)):
    # Ensure the days is a multiple of 15, because we are predicting in blocks of 15 days
    if days % 15 != 0:
        raise HTTPException(status_code=400, detail="The number of days must be a multiple of 15.")

    product_ids = data['product_id'].unique()
    all_predictions = []

    for product_id in product_ids:
        product_predictions = predict_for_15_day_intervals(product_id, days)
        all_predictions.extend(product_predictions)

    if not all_predictions:
        raise HTTPException(status_code=400, detail="No predictions could be generated.")

    pd.DataFrame(all_predictions).to_csv("inventory_predictions.csv", index=False)
    return all_predictions

# GET endpoint to make daily predictions
@app.get("/predict-daily/")
def predict_daily_endpoint():
    product_ids = data['product_id'].unique()
    all_predictions = []

    for product_id in product_ids:
        product_predictions = predict_for_daily_intervals(product_id)
        all_predictions.extend(product_predictions)

    if not all_predictions:
        raise HTTPException(status_code=400, detail="No daily predictions could be generated.")

    pd.DataFrame(all_predictions).to_csv("daily_inventory_predictions.csv", index=False)
    return all_predictions
