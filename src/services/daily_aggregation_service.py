from collections import defaultdict, Counter
from statistics import mean
from datetime import datetime


class AggregationService:
    def aggregate_daily_forecast(self, hourly_forecast: list, ml_engine):
        grouped = defaultdict(list)

        for item in hourly_forecast:
            # item içinde "hour" ve "date" olduğunu varsayıyoruz
            # weather_service içinde bunları eklemiş olman lazım
            if 8 <= item["hour"] <= 23:
                grouped[item["date"]].append(item)

        daily_results = []

        for date, day_hours in grouped.items():
            umbrella_preds = []
            clothing_preds = []
            suitability_scores = []
            hourly_details = []

            for hour_data in day_hours:
                hour = hour_data["hour"]
                month = int(hour_data["date"].split("-")[1])

                analysis = ml_engine.predict_all(
                    weather_data=hour_data,
                    hour=hour,
                    month=month
                )

                umbrella_pred = analysis["umbrella"]["prediction"]
                clothing_pred = analysis["clothing"]["prediction"]
                suitability_pred = analysis["suitability"]["prediction"]

                umbrella_preds.append(bool(umbrella_pred))
                clothing_preds.append(clothing_pred)
                suitability_scores.append(float(suitability_pred))

                hourly_details.append({
                    "datetime": hour_data["datetime"],
                    "hour": hour,
                    "umbrella": bool(umbrella_pred),
                    "clothing": clothing_pred,
                    "suitability_score": float(suitability_pred)
                })
                weather_codes = [x.get("weather_condition_code", 0) for x in day_hours]
                general_condition = self._get_weather_description(weather_codes)

            weather_summary = {
                "date": date,
                "general_condition": general_condition,
                "min_temperature_c": round(min(x["temperature_c"] for x in day_hours), 1),
                "max_temperature_c": round(max(x["temperature_c"] for x in day_hours), 1),
                "avg_feels_like_c": round(mean(x["feels_like_c"] for x in day_hours), 1),
                "total_precipitation_mm": round(sum(x["precipitation_mm"] for x in day_hours), 1),
                "avg_wind_speed_kmh": round(mean(x["wind_speed_kmh"] for x in day_hours), 1),
                "max_wind_gust_kmh": round(max(x["wind_gust_kmh"] for x in day_hours), 1),
                "avg_humidity_pct": round(mean(x["humidity_pct"] for x in day_hours), 1),
                "avg_cloud_cover_pct": round(mean(x["cloud_cover_pct"] for x in day_hours), 1),
                "avg_uv_index": round(mean(x.get("uv_index", 0) for x in day_hours), 1),
            }

            analysis_summary = {
                "umbrella_needed": any(umbrella_preds),
                "average_suitability_score": round(mean(suitability_scores), 2) if suitability_scores else None,
                "dominant_clothing": Counter(clothing_preds).most_common(1)[0][0] if clothing_preds else None,
            }

            daily_results.append({
                "date": date,
                "weather_summary": weather_summary,
                "analysis_summary": analysis_summary,
                "hourly_details": hourly_details
            })

        return daily_results
    
    def _get_weather_description(self, weather_codes: list) -> str:
        code_map = {
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

        labels = [code_map.get(code, "clear") for code in weather_codes]
        return Counter(labels).most_common(1)[0][0]
    
    