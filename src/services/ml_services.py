#Open meteo'dan gelen cevabı parse edecek, modelinin beklediği forma çevirecek
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd


class MLService:
    def __init__(self):
        base_path = Path(__file__).resolve().parents[2] / "models"

        # Ana modeller
        self.clothing_model = joblib.load(base_path / "clothing_model.pkl")
        self.umbrella_model = joblib.load(base_path / "umbrella_model.pkl")
        self.suitability_model = joblib.load(base_path / "suitability_model.pkl")

        # Confidence modelleri
        self.clothing_confidence_model = joblib.load(base_path / "clothing_confidence_model.pkl")
        self.umbrella_confidence_model = joblib.load(base_path / "umbrella_confidence_model.pkl")
        self.condition_confidence_model = joblib.load(base_path / "condition_confidence_model.pkl")

        # Encoder
        self.clothing_label_encoder = joblib.load(base_path / "clothing_label_encoder.pkl")

        self.clothing_feature_order = [
            "temperature_c",
            "wind_speed_kmh",
            "wind_gust_kmh",
            "uv_index",
            "hour_of_day",
            "month",
        ]

        self.umbrella_feature_order = [
            "temperature_c",
            "feels_like_c",
            "dew_point_c",
            "humidity_pct",
            "pressure_hpa",
            "wind_speed_kmh",
            "wind_direction_deg",
            "wind_gust_kmh",
            "cloud_cover_pct",
            "visibility_km",
            "uv_index",
            "hour_of_day",
            "month",
            "season",
        ]

        self.suitability_feature_order = [
            "temperature_c",
            "feels_like_c",
            "dew_point_c",
            "humidity_pct",
            "pressure_hpa",
            "wind_speed_kmh",
            "wind_direction_deg",
            "wind_gust_kmh",
            "cloud_cover_pct",
            "visibility_km",
            "uv_index",
            "hour_of_day",
            "month",
        ]

    def _to_dataframe(self, features: dict, feature_order: list[str]) -> pd.DataFrame:
        return pd.DataFrame([[features[col] for col in feature_order]], columns=feature_order)

    def _to_python_type(self, value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        return "autumn"

    def _get_weather_description(self, code: int) -> str:
        mapping = {
            0: "clear",
            1: "clear",
            2: "partly_cloudy",
            3: "cloudy",
            45: "cloudy",
            51: "drizzle",
            53: "drizzle",
            55: "drizzle",
            61: "rain",
            63: "rain",
            65: "heavy_rain",
            71: "snow",
            73: "snow",
            75: "heavy_snow",
            77: "snow",
            80: "rain",
            81: "rain",
            82: "heavy_rain",
            85: "snow",
            86: "heavy_snow",
            95: "thunderstorm",
            96: "thunderstorm",
            99: "thunderstorm",
        }
        return mapping.get(code, "clear")

    def _extract_clothing_features(self, weather_data: dict, hour: int, month: int):
        return {
            "temperature_c": weather_data["temperature_c"],
            "wind_speed_kmh": weather_data["wind_speed_kmh"],
            "wind_gust_kmh": weather_data["wind_gust_kmh"],
            "uv_index": weather_data.get("uv_index", 0),
            "hour_of_day": hour,
            "month": month,
        }

    def _extract_umbrella_features(self, weather_data: dict, hour: int, month: int):
        return {
            "temperature_c": weather_data["temperature_c"],
            "feels_like_c": weather_data["feels_like_c"],
            "dew_point_c": weather_data["dew_point_c"],
            "humidity_pct": weather_data["humidity_pct"],
            "pressure_hpa": weather_data["pressure_hpa"],
            "wind_speed_kmh": weather_data["wind_speed_kmh"],
            "wind_direction_deg": weather_data["wind_direction_deg"],
            "wind_gust_kmh": weather_data["wind_gust_kmh"],
            "cloud_cover_pct": weather_data["cloud_cover_pct"],
            "visibility_km": weather_data["visibility_km"],
            "uv_index": weather_data.get("uv_index", 0),
            "hour_of_day": hour,
            "month": month,
            "season": self._get_season(month),
        }

    def _extract_suitability_features(self, weather_data: dict, hour: int, month: int):
        return {
            "temperature_c": weather_data["temperature_c"],
            "feels_like_c": weather_data["feels_like_c"],
            "dew_point_c": weather_data["dew_point_c"],
            "humidity_pct": weather_data["humidity_pct"],
            "pressure_hpa": weather_data["pressure_hpa"],
            "wind_speed_kmh": weather_data["wind_speed_kmh"],
            "wind_direction_deg": weather_data["wind_direction_deg"],
            "wind_gust_kmh": weather_data["wind_gust_kmh"],
            "cloud_cover_pct": weather_data["cloud_cover_pct"],
            "visibility_km": weather_data["visibility_km"],
            "uv_index": weather_data.get("uv_index", 0),
            "hour_of_day": hour,
            "month": month,
        }

    def _hour_bucket(self, hour: int) -> str:
        if 5 <= hour < 11:
            return "morning"
        elif 11 <= hour < 17:
            return "midday"
        elif 17 <= hour < 22:
            return "evening"
        return "night"

    def _lead_bucket(self, lead_time_hours: int) -> str:
        if lead_time_hours <= 1:
            return "h1"
        elif lead_time_hours <= 3:
            return "h3"
        elif lead_time_hours <= 6:
            return "h6"
        elif lead_time_hours <= 12:
            return "h12"
        return "h24"

    def _confidence_label(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        elif score >= 0.50:
            return "medium"
        return "low"

    def _normalize_confidence(self, score: float, lead_time: int, task: str) -> float:
        score = 0.5 + (score - 0.5) * 0.6

        if task == "umbrella":
            penalty = 1 - 0.004 * lead_time
        elif task == "clothing":
            penalty = 1 - 0.008 * lead_time
        else:
            penalty = 1 - 0.008 * lead_time

        score *= penalty
        score = max(0.05, min(0.95, score))
        return round(score, 4)

    def _derive_lead_time_hours(self, target_hour: int, issue_hour: int) -> int:
        lead = target_hour - issue_hour
        if lead <= 0:
            lead += 24
        return lead

    def _build_confidence_features(
        self,
        weather_data: dict,
        forecasted_umbrella: bool,
        forecasted_clothing: str,
        hour: int,
        month: int,
    ) -> dict:
        now = datetime.now()
        issue_hour = now.hour
        issue_month = now.month

        target_hour = hour
        target_month = month

        lead_time_hours = self._derive_lead_time_hours(target_hour, issue_hour)

        forecasted_temp_c = float(weather_data["temperature_c"])
        forecasted_precip_mm = float(weather_data["precipitation_mm"])
        forecasted_wind_kmh = float(weather_data["wind_speed_kmh"])

        return {
            "lead_time_hours": lead_time_hours,
            "forecasted_temp_c": forecasted_temp_c,
            "forecasted_precip_mm": forecasted_precip_mm,
            "forecasted_wind_kmh": forecasted_wind_kmh,
            "forecasted_umbrella": int(bool(forecasted_umbrella)),
            "forecasted_clothing": str(forecasted_clothing),
            "target_hour": int(target_hour),
            "target_month": int(target_month),
            "issue_hour": int(issue_hour),
            "issue_month": int(issue_month),
            "is_rainy_forecast": int(forecasted_precip_mm > 0.0),
            "is_windy_forecast": int(forecasted_wind_kmh >= 20.0),
            "is_cold": int(forecasted_temp_c <= 12.0),
            "is_hot": int(forecasted_temp_c >= 28.0),
            "precip_wind_interaction": float(forecasted_precip_mm * forecasted_wind_kmh),
            "temp_wind_interaction": float(forecasted_temp_c * forecasted_wind_kmh),
            "lead_time_bucket": self._lead_bucket(lead_time_hours),
            "target_period": self._hour_bucket(target_hour),
            "issue_period": self._hour_bucket(issue_hour),
        }

    def _predict_confidences(
        self,
        weather_data: dict,
        forecasted_umbrella: bool,
        forecasted_clothing: str,
        hour: int,
        month: int,
    ) -> dict:
        feature_dict = self._build_confidence_features(
            weather_data=weather_data,
            forecasted_umbrella=forecasted_umbrella,
            forecasted_clothing=forecasted_clothing,
            hour=hour,
            month=month,
        )

        sample = pd.DataFrame([feature_dict])
        lead_time = int(feature_dict["lead_time_hours"])

        umbrella_raw = float(self.umbrella_confidence_model.predict_proba(sample)[0, 1])
        clothing_raw = float(self.clothing_confidence_model.predict_proba(sample)[0, 1])
        condition_raw = float(self.condition_confidence_model.predict_proba(sample)[0, 1])

        umbrella_score = self._normalize_confidence(umbrella_raw, lead_time, "umbrella")
        clothing_score = self._normalize_confidence(clothing_raw, lead_time, "clothing")
        condition_score = self._normalize_confidence(condition_raw, lead_time, "condition")

        return {
            "umbrella_confidence": {
                "score": float(umbrella_score),
                "label": str(self._confidence_label(umbrella_score)),
            },
            "clothing_confidence": {
                "score": float(clothing_score),
                "label": str(self._confidence_label(clothing_score)),
            },
            "condition_confidence": {
                "score": float(condition_score),
                "label": str(self._confidence_label(condition_score)),
            },
        }

    def predict_all(self, weather_data: dict, hour: int, month: int):
        # Clothing
        clothing_features = self._extract_clothing_features(weather_data, hour, month)
        X_clothing = self._to_dataframe(clothing_features, self.clothing_feature_order)
        clothing_pred_raw = self.clothing_model.predict(X_clothing)[0]
        clothing_pred_raw = self._to_python_type(clothing_pred_raw)

        try:
            clothing_pred = self.clothing_label_encoder.inverse_transform([clothing_pred_raw])[0]
        except Exception:
            clothing_pred = clothing_pred_raw

        clothing_pred = self._to_python_type(clothing_pred)
        clothing_pred = str(clothing_pred)

        # Umbrella
        umbrella_features = self._extract_umbrella_features(weather_data, hour, month)
        X_umbrella = self._to_dataframe(umbrella_features, self.umbrella_feature_order)
        umbrella_pred_raw = self.umbrella_model.predict(X_umbrella)[0]
        umbrella_pred_raw = self._to_python_type(umbrella_pred_raw)

        if isinstance(umbrella_pred_raw, bool):
            umbrella_pred = umbrella_pred_raw
        elif isinstance(umbrella_pred_raw, (int, float)):
            umbrella_pred = bool(umbrella_pred_raw)
        else:
            umbrella_pred = bool(umbrella_pred_raw)

        umbrella_pred = bool(self._to_python_type(umbrella_pred))

        # Suitability
        suitability_features = self._extract_suitability_features(weather_data, hour, month)
        X_suitability = self._to_dataframe(suitability_features, self.suitability_feature_order)
        suitability_pred_raw = self.suitability_model.predict(X_suitability)[0]
        suitability_pred_raw = self._to_python_type(suitability_pred_raw)

        try:
            suitability_pred = float(suitability_pred_raw)
        except Exception:
            suitability_pred = suitability_pred_raw

        suitability_pred = self._to_python_type(suitability_pred)

        # Confidence
        confidences = self._predict_confidences(
            weather_data=weather_data,
            forecasted_umbrella=umbrella_pred,
            forecasted_clothing=clothing_pred,
            hour=hour,
            month=month,
        )

        return {
            "clothing": {
                "prediction": str(clothing_pred),
                "confidence": {
                    "score": float(confidences["clothing_confidence"]["score"]),
                    "label": str(confidences["clothing_confidence"]["label"]),
                },
            },
            "umbrella": {
                "prediction": bool(umbrella_pred),
                "confidence": {
                    "score": float(confidences["umbrella_confidence"]["score"]),
                    "label": str(confidences["umbrella_confidence"]["label"]),
                },
            },
            "suitability": {
                "prediction": float(suitability_pred),
                "confidence": {
                    "score": float(confidences["condition_confidence"]["score"]),
                    "label": str(confidences["condition_confidence"]["label"]),
                },
            },
        }