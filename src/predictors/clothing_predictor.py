#Sadece clothing modeliyle ilgilenir, tahmini alır. raw advice alır, model inference yapar
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "models", "clothing_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "clothing_label_encoder.pkl")

clothing_model = joblib.load(MODEL_PATH)
clothing_label_encoder = joblib.load(ENCODER_PATH)

def predict_clothing(features: dict) -> dict:
    model_input = pd.DataFrame([{
        "temperature_c": features["temperature_c"],
        "wind_speed_kmh": features["wind_speed_kmh"],
        "wind_gust_kmh": features["wind_gust_kmh"],
        "uv_index": features["uv_index"],
        "hour_of_day": features["hour_of_day"],
        "month": features["month"]
    }])

    pred_encoded = int(clothing_model.predict(model_input)[0])
    pred_label = clothing_label_encoder.inverse_transform([pred_encoded])[0]

    return {
        "clothing_recommendation": str(pred_label)
    }