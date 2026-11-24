from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

umbrella_conf_model = joblib.load(MODEL_DIR / "../../models/umbrella_confidence_model.pkl")
clothing_conf_model = joblib.load(MODEL_DIR / "../../models/clothing_confidence_model.pkl")
condition_conf_model = joblib.load(MODEL_DIR / "../../models/condition_confidence_model.pkl")

def derive_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"
    
    
def hour_bucket(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 17:
        return "midday"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def lead_bucket(lead_time_hours: int) -> str:
    if lead_time_hours <= 1:
        return "h1"
    elif lead_time_hours <= 3:
        return "h3"
    elif lead_time_hours <= 6:
        return "h6"
    elif lead_time_hours <= 12:
        return "h12"
    else:
        return "h24"


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    elif score >= 0.50:
        return "medium"
    else:
        return "low"


def normalize_confidence(score: float, lead_time: int, task: str) -> float:
    score = 0.5 + (score - 0.5) * 0.6

    if task == "umbrella":
        penalty = 1 - 0.004 * lead_time
    elif task == "clothing":
        penalty = 1 - 0.008 * lead_time
    else:  # condition
        penalty = 1 - 0.008 * lead_time

    score *= penalty
    score = max(0.05, min(0.95, score))

    return round(score, 4)


def derive_lead_time_hours(target_hour: int, issue_hour: int) -> int:
    """
    Assumption:
    target hour is the next occurrence of that hour within 24 hours.
    """
    lead = target_hour - issue_hour
    if lead <= 0:
        lead += 24
    return lead


def derive_confidence_base_features(features: dict) -> dict:
    """
    Expected input:
    - Weather24hInput fields
    - forecasted_umbrella
    - forecasted_clothing
    """
    now = datetime.now()

    issue_hour = now.hour
    issue_month = now.month

    target_hour = int(features["hour_of_day"])
    target_month = int(features["month"])

    lead_time_hours = derive_lead_time_hours(
        target_hour=target_hour,
        issue_hour=issue_hour,
    )

    return {
        "lead_time_hours": lead_time_hours,
        "forecasted_temp_c": float(features["temperature_c"]),
        "forecasted_precip_mm": float(features["precip_mm"]),
        "forecasted_wind_kmh": float(features["wind_speed_kmh"]),
        "forecasted_umbrella": int(features["forecasted_umbrella"]),
        "forecasted_clothing": str(features["forecasted_clothing"]),
        "target_hour": target_hour,
        "target_month": target_month,
        "issue_hour": issue_hour,
        "issue_month": issue_month,
    }


def build_confidence_features(features: dict) -> dict:
    base = derive_confidence_base_features(features)

    lead_time_hours = int(base["lead_time_hours"])
    forecasted_temp_c = float(base["forecasted_temp_c"])
    forecasted_precip_mm = float(base["forecasted_precip_mm"])
    forecasted_wind_kmh = float(base["forecasted_wind_kmh"])
    forecasted_umbrella = int(base["forecasted_umbrella"])
    forecasted_clothing = str(base["forecasted_clothing"])
    target_hour = int(base["target_hour"])
    target_month = int(base["target_month"])
    issue_hour = int(base["issue_hour"])
    issue_month = int(base["issue_month"])

    return {
        "lead_time_hours": lead_time_hours,
        "forecasted_temp_c": forecasted_temp_c,
        "forecasted_precip_mm": forecasted_precip_mm,
        "forecasted_wind_kmh": forecasted_wind_kmh,
        "forecasted_umbrella": forecasted_umbrella,
        "forecasted_clothing": forecasted_clothing,
        "target_hour": target_hour,
        "target_month": target_month,
        "issue_hour": issue_hour,
        "issue_month": issue_month,
        "is_rainy_forecast": int(forecasted_precip_mm > 0.0),
        "is_windy_forecast": int(forecasted_wind_kmh >= 20.0),
        "is_cold": int(forecasted_temp_c <= 12.0),
        "is_hot": int(forecasted_temp_c >= 28.0),
        "precip_wind_interaction": forecasted_precip_mm * forecasted_wind_kmh,
        "temp_wind_interaction": forecasted_temp_c * forecasted_wind_kmh,
        "lead_time_bucket": lead_bucket(lead_time_hours),
        "target_period": hour_bucket(target_hour),
        "issue_period": hour_bucket(issue_hour),
    }


def predict_confidence(features: dict) -> dict:
    """
    features must include:
    - Weather24hInput fields
    - forecasted_umbrella
    - forecasted_clothing
    """
    feature_dict = build_confidence_features(features)
    sample = pd.DataFrame([feature_dict])

    lead_time = int(feature_dict["lead_time_hours"])

    umbrella_raw = float(umbrella_conf_model.predict_proba(sample)[0, 1])
    clothing_raw = float(clothing_conf_model.predict_proba(sample)[0, 1])
    condition_raw = float(condition_conf_model.predict_proba(sample)[0, 1])

    umbrella_score = normalize_confidence(umbrella_raw, lead_time, "umbrella")
    clothing_score = normalize_confidence(clothing_raw, lead_time, "clothing")
    condition_score = normalize_confidence(condition_raw, lead_time, "condition")

    return {
        "umbrella_confidence": {
            "score": umbrella_score,
            "label": confidence_label(umbrella_score),
        },
        "clothing_confidence": {
            "score": clothing_score,
            "label": confidence_label(clothing_score),
        },
        "condition_confidence": {
            "score": condition_score,
            "label": confidence_label(condition_score),
        },
    }