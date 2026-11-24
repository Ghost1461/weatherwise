import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "umbrella_model.pkl")

umbrella_model = joblib.load(MODEL_PATH)

def predict_umbrella(features: dict) -> dict:
    model_input = pd.DataFrame([{
        "temperature_c": features["temperature_c"],
        "feels_like_c": features["feels_like_c"],
        "dew_point_c": features["dew_point_c"],
        "humidity_pct": features["humidity_pct"],
        "pressure_hpa": features["pressure_hpa"],
        "wind_speed_kmh": features["wind_speed_kmh"],
        "wind_direction_deg": features["wind_direction_deg"],
        "wind_gust_kmh": features["wind_gust_kmh"],
        "cloud_cover_pct": features["cloud_cover_pct"],
        "visibility_km": features["visibility_km"],
        "uv_index": features["uv_index"],
        "hour_of_day": features["hour_of_day"],
        "month": features["month"],
        "season": features["season"],
    }])

    print("TYPE:", type(model_input))
    print(model_input)

    prediction = int(umbrella_model.predict(model_input)[0])
    advice = "Take an umbrella." if prediction == 1 else "No umbrella needed."

    return {
        "umbrella_needed": int(prediction),
        "umbrella_advice": str(advice)
    }