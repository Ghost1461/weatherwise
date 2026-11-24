# activity_service.py

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


class ActivityService:
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
        "hour_of_day",
    ]

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent.parent
        model_dir = base_dir / "models"

        self.activity_models = joblib.load(model_dir / "activity_models.pkl")
        self.go_model = joblib.load(model_dir / "go_model.pkl")
        self.activity_thresholds = joblib.load(model_dir / "activity_thresholds.pkl")
        self.activity_thresholds = {
            k: float(v) for k, v in self.activity_thresholds.items()
        }

    def _get_weather_description(self, code) -> str:
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

        if code is None:
            return "clear"

        return mapping.get(int(code), "clear")

    def _get_season(self, month: int) -> str:
        if month in (12, 1, 2):
            return "winter"
        if month in (3, 4, 5):
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        return "autumn"


    def build_activity_input(self, weather: dict, dt: datetime) -> dict:
        return {
            "month": dt.month,
            "is_weekend": 1 if dt.weekday() >= 5 else 0,
            "season": self._get_season(dt.month),
            "weather_condition": self._get_weather_description(weather.get("weather_code")),
            "temperature_c": weather.get("temperature_c"),
            "feels_like_c": weather.get("feels_like_c"),
            "humidity_pct": weather.get("humidity_pct"),
            "wind_speed_kmh": weather.get("wind_speed_kmh"),
            "precipitation_mm": weather.get("precipitation_mm"),
            "cloud_cover_pct": weather.get("cloud_cover_pct"),
            "uv_index": weather.get("uv_index"),
            "hour_of_day": dt.hour,
        }

    def build_activity_input_df(self, features: dict) -> pd.DataFrame:
        row = {key: features[key] for key in self.ACTIVITY_FEATURES}
        return pd.DataFrame([row])[self.ACTIVITY_FEATURES]

    def rank_activities_for_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        input_df = input_df[self.ACTIVITY_FEATURES].copy()

        go_pred = int(self.go_model.predict(input_df)[0])
        go_prob = float(self.go_model.predict_proba(input_df)[0][1])
        gate = 0.3 + 0.7 * go_prob

        rain = float(input_df["precipitation_mm"].iloc[0])
        wind = float(input_df["wind_speed_kmh"].iloc[0])
        hour = int(input_df["hour_of_day"].iloc[0])
        temp = float(input_df["temperature_c"].iloc[0])

        results = []

        for activity, model in self.activity_models.items():
            activity_prob = float(model.predict_proba(input_df)[0][1])
            threshold = float(self.activity_thresholds.get(activity, 0.5))

            relative_activity_score = (
                activity_prob / threshold if threshold > 0 else activity_prob
            )
            relative_activity_score = min(relative_activity_score, 1.5)

            base_score = (
                0.85 * activity_prob +
                0.15 * (relative_activity_score / 1.5)
            ) * gate

            adjustment = 1.0

            if activity == "walking":
                if rain < 2 and wind < 20 and hour < 23:
                    adjustment = 1.7
                elif rain >= 5 or wind >= 25 or hour >= 23:
                    adjustment = 1.1
                else:
                    adjustment = 1.0

            elif activity == "cycling":
                if rain == 0 and wind < 20 and hour < 21:
                    adjustment = 1.3
                elif rain < 2 and wind < 18 and hour < 21:
                    adjustment = 1.2
                elif rain >= 3 or wind >= 20 or hour >= 21:
                    adjustment = 0.4
                else:
                    adjustment = 1.0

            elif activity == "driving":
                if rain >= 4 or hour >= 21:
                    adjustment = 1.2
                else:
                    adjustment = 0.6

            elif activity == "running":
                if rain >= 2 or hour >= 21:
                    adjustment = 0.6
                elif rain == 0 and wind < 18 and 10 <= hour <= 19:
                    adjustment = 1.1
                else:
                    adjustment = 1.0

            elif activity == "picnic":
                if hour >= 20:
                    adjustment = 0.25
                elif rain >= 2:
                    adjustment = 0.5
                elif rain == 0 and 15 <= temp <= 28:
                    adjustment = 1.25
                else:
                    adjustment = 1.0

            elif activity == "commute":
                adjustment = 1.0

            final_score = base_score * adjustment

            results.append({
                "activity": activity,
                "activity_probability": round(activity_prob, 4),
                "threshold": round(threshold, 4),
                "relative_activity_score": round(relative_activity_score, 4),
                "go_or_no_pred": go_pred,
                "go_or_no_probability": round(go_prob, 4),
                "gate": round(gate, 4),
                "adjustment": round(adjustment, 2),
                "final_score": round(final_score, 4),
            })

        return pd.DataFrame(results).sort_values(
            by="final_score",
            ascending=False
        ).reset_index(drop=True)

    def predict_activity_recommendations(self, weather: dict, dt: datetime) -> dict:
        activity_input = self.build_activity_input(weather, dt)
        input_df = self.build_activity_input_df(activity_input)
        ranked_df = self.rank_activities_for_input(input_df)

        return {
            "go_or_no_pred": int(ranked_df.iloc[0]["go_or_no_pred"]) if not ranked_df.empty else 0,
            "go_or_no_probability": float(ranked_df.iloc[0]["go_or_no_probability"]) if not ranked_df.empty else 0.0,
            "activities": ranked_df.to_dict(orient="records"),
        }

    def get_top_activities(self, weather: dict, dt: datetime, top_k: int = 3) -> list[dict]:
        prediction = self.predict_activity_recommendations(weather, dt)
        activities = prediction["activities"]

        filtered = [
            item for item in activities
            if item["final_score"] >= 0.35
        ]

        return filtered[:top_k]