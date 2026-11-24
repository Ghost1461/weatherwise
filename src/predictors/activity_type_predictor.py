import json
import joblib
import pandas as pd
from pathlib import Path


ACTIVITY_FEATURES = [
    "month",
    "is_weekend",
    "season",
    "weather_condition",
    "temperature_c",
    "feels_like_c",
    "humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
    "cloud_cover_pct",
    "uv_index",
    "hour_of_day"
]

# =========================
# LOAD TRAINED ARTIFACTS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

activity_models = joblib.load(MODEL_DIR / "activity_models.pkl")
go_model = joblib.load(MODEL_DIR / "go_model.pkl")
activity_thresholds = joblib.load(MODEL_DIR / "activity_thresholds.pkl")

activity_thresholds = {
    k: float(v) for k, v in activity_thresholds.items()
}

def build_activity_input_df(features: dict) -> pd.DataFrame:
    row = {
        "month": features["month"],
        "is_weekend": features["is_weekend"],
        "season": features["season"],
        "weather_condition": features["weather_condition"],
        "temperature_c": features["temperature_c"],
        "feels_like_c": features["feels_like_c"],
        "humidity_pct": features["humidity_pct"],
        "wind_speed_kmh": features["wind_speed_kmh"],
        "precipitation_mm": features["precipitation_mm"],
        "cloud_cover_pct": features["cloud_cover_pct"],
        "uv_index": features["uv_index"],
        "hour_of_day": features["hour_of_day"],
    }

    return pd.DataFrame([row])[ACTIVITY_FEATURES]


def rank_activities_for_input(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df = input_df[ACTIVITY_FEATURES].copy()

    go_pred = int(go_model.predict(input_df)[0])
    go_prob = float(go_model.predict_proba(input_df)[0][1])

    gate = 0.3 + 0.7 * go_prob

    rain = float(input_df["precipitation_mm"].iloc[0])
    wind = float(input_df["wind_speed_kmh"].iloc[0])
    hour = int(input_df["hour_of_day"].iloc[0])
    temp = float(input_df["temperature_c"].iloc[0])

    results = []

    for activity, model in activity_models.items():
        activity_prob = float(model.predict_proba(input_df)[0][1])
        threshold = float(activity_thresholds.get(activity, 0.5))

        relative_activity_score = activity_prob / threshold if threshold > 0 else activity_prob
        relative_activity_score = min(relative_activity_score, 1.5)

        base_score = (
            0.85 * activity_prob +
            0.15 * (relative_activity_score / 1.5)
        ) * gate

        adjustment = 1.0

        # walking: iyi havada artsın
        if activity == "walking":
            if rain < 2 and wind < 20 and hour < 23:
                adjustment = 1.7
            elif rain >= 5 or wind >= 25 or hour >= 23:
                adjustment = 1.1
            else:
                adjustment = 1.00

        # cycling: iyi havada artsın, kötü havada düşsün
        elif activity == "cycling":
            if rain == 0 and wind < 20 and hour < 21:
                adjustment = 1.3
            elif rain < 2 and wind < 18 and hour < 21:
                adjustment = 1.2
            elif rain >= 3 or wind >= 20 or hour >= 21:
                adjustment = 0.4
            else:
                adjustment = 1.00

        # driving: genel olarak biraz düşsün, ama kötü havada/gecede avantajlı olsun
        elif activity == "driving":
            if rain >= 4 or hour >= 21:
                adjustment = 1.2
            else:
                adjustment = 0.60

        # running: yağmurda ve gecede düşsün
        elif activity == "running":
            if rain >= 2 or hour >= 21:
                adjustment = 0.60
            elif rain == 0 and wind < 18 and 10 <= hour <= 19:
                adjustment = 1.10
            else:
                adjustment = 1.00

        # picnic: gecede düşsün, iyi havada artsın
        elif activity == "picnic":
            if hour >= 20:
                adjustment = 0.25
            elif rain >= 2:
                adjustment = 0.50
            elif rain == 0 and 15 <= temp <= 28:
                adjustment = 1.25
            else:
                adjustment = 1.00

        elif activity == "commute":
            adjustment = 1.00

        final_score = base_score * adjustment

        results.append({
            "activity": activity,
            "activity_probability": round(activity_prob, 4),
            "threshold": round(threshold, 2),
            "relative_activity_score": round(relative_activity_score, 4),
            "go_or_no_pred": go_pred,
            "go_or_no_probability": round(go_prob, 4),
            "gate": round(gate, 4),
            "adjustment": round(adjustment, 2),
            "final_score": round(final_score, 4)
        })

    results_df = pd.DataFrame(results).sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    return results_df


def predict_activity_recommendations(activity_input):
    input_df = build_activity_input_df(activity_input)
    ranked_df = rank_activities_for_input(input_df)

    return {
        "go_or_no_pred": int(ranked_df.iloc[0]["go_or_no_pred"]) if not ranked_df.empty else 0,
        "go_or_no_probability": float(ranked_df.iloc[0]["go_or_no_probability"]) if not ranked_df.empty else 0.0,
        "activities": ranked_df.to_dict(orient="records")
    }