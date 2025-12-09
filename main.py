from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

model = load_model("rul_lstm_model.keras")
scaler = joblib.load("scaler.save")

app = FastAPI()

SEQ_LEN = 24  # Using past 24 time steps for prediction



engineered_features = [
    # Raw signals
    'temperature', 'vibration', 'pressure',

    # Rolling means (short-term smoothing)
    'temperature_roll_mean', 'vibration_roll_mean', 'pressure_roll_mean',

    # Lag features (temporal dependencies)
    'temperature_lag1', 'vibration_lag1', 'pressure_lag1',
    'temperature_lag2', 'vibration_lag2', 'pressure_lag2',
    'temperature_lag3', 'vibration_lag3', 'pressure_lag3',

    # Variability (very predictive for failures)
    'temperature_roll_std', 'vibration_roll_std', 'pressure_roll_std',

    # Long-window degradation indicators
    'vibration_min', 'vibration_max', 'vibration_std', 'vibration_slope',

    # Events
    'failure_event'
]

class EquipmentData(BaseModel):
    equipment_id: int
    timestamp: str
    temperature: float
    vibration: float
    pressure: float
    failure_event: int = 0

@app.post("/predict_rul_all")
def predict_rul_all(data: list[EquipmentData]):
    df = pd.DataFrame([d.model_dump() for d in data])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    predictions = {}

    for eq in df['equipment_id'].unique():
        eq_df = df[df['equipment_id'] == eq].copy()
        eq_df = eq_df.sort_values("timestamp")

        # Feature engineering
        eq_df["temperature_roll_mean"] = eq_df["temperature"].rolling(3, min_periods=1).mean()
        eq_df["vibration_roll_mean"] = eq_df["vibration"].rolling(3, min_periods=1).mean()
        eq_df["pressure_roll_mean"] = eq_df["pressure"].rolling(3, min_periods=1).mean()

        eq_df["temperature_roll_std"] = eq_df["temperature"].rolling(3, min_periods=1).std().fillna(0)
        eq_df["vibration_roll_std"] = eq_df["vibration"].rolling(3, min_periods=1).std().fillna(0)
        eq_df["pressure_roll_std"] = eq_df["pressure"].rolling(3, min_periods=1).std().fillna(0)


        for lag in [1, 2, 3]:
            eq_df[f"temperature_lag{lag}"] = eq_df["temperature"].shift(lag).fillna(eq_df.mean())
            eq_df[f"vibration_lag{lag}"] = eq_df["vibration"].shift(lag).fillna(eq_df.mean())
            eq_df[f"pressure_lag{lag}"] = eq_df["pressure"].shift(lag).fillna(eq_df.mean())
        
        eq_df['vibration_slope'] = eq_df.groupby('equipment_id')['vibration_roll_mean'].transform(lambda x: x.diff().fillna(0))
        eq_df["vibration_min"] = eq_df["vibration_roll_mean"].rolling(24, min_periods=1).min()
        eq_df["vibration_max"] = eq_df["vibration_roll_mean"].rolling(24, min_periods=1).max()
        eq_df["vibration_std"] = eq_df["vibration_roll_mean"].rolling(24, min_periods=1).std().fillna(0)

        # Scaling
        df_scaled = scaler.transform(eq_df[engineered_features])

        # Sequence extraction
        if len(df_scaled) < SEQ_LEN:
            predictions[int(eq)] = {"error": f"Need {SEQ_LEN} rows, got {len(df_scaled)}"}
            continue

        seq = df_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(engineered_features))
        prediction = float(model.predict(seq)[0][0])

        predictions[int(eq)] = {"predicted_RUL": float(prediction)}

    return jsonable_encoder(predictions)
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)