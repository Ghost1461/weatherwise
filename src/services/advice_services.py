# advice_service.py

from datetime import datetime
import requests
from .weather_service import WeatherService
from src.services.ml_services import MLService
from .llm_service import LLMService
from .prompt_builder import PromptBuilderService
from .activity_service import ActivityService
from src.schemas.open_meteo_schema import WeatherRequest, OpenMeteoParams
from datetime import datetime, timedelta
from src.services.daily_aggregation_service import AggregationService


class AdviceService:
    def __init__(self):
        self.weather_engine = WeatherService()
        self.ml_engine = MLService()
        self.prompt_builder = PromptBuilderService()
        self.llm_engine = LLMService()
        self.activity_engine = ActivityService()
        self.aggregation_service = AggregationService()

    async def fetch_weather(self, latitude: float, longitude: float):
        """Sadece koordinat alır, OpenMeteoParams'ı içerde oluşturup işi bitirir."""
        om_params = OpenMeteoParams(latitude=latitude, longitude=longitude)
        return await self.weather_engine.get_formatted_weather(
            om_params.latitude,
            om_params.longitude
        )

    def decide_tone(self, user_request: WeatherRequest) -> str:
        if not user_request.personalization_on:
            return "formal"

        if user_request.tone == "friendly":
            return "friendly"

        if user_request.tone == "formal":
            return "formal"

        return "formal"

    def _get_effective_preferences(self, user_request: WeatherRequest) -> dict:
        personalization_on = user_request.personalization_on
        tone = self.decide_tone(user_request)

        if not personalization_on:
            return {
                "personalization_on": False,
                "has_dog": None,
                "has_car": None,
                "near_sea": None,
                "sun_allergy": None,
                "tone": tone,
            }

        return {
            "personalization_on": True,
            "has_dog": user_request.has_dog,
            "has_car": user_request.has_car,
            "near_sea": user_request.near_sea,
            "sun_allergy": user_request.sun_allergy,
            "tone": tone,
        }


    from datetime import datetime, timedelta


    async def get_final_recommendation(self, user_request: WeatherRequest):
        city_name, lat, lon = self._resolve_coordinates(user_request)
        weather_full = await self.fetch_weather(lat, lon)

        effective_prefs = self._get_effective_preferences(user_request)

        hourly_forecast = weather_full["hourly_forecast"]
        now = datetime.now()

        results = []

        # Şu anki saate en yakın index
        start_index = now.hour

        # 4 saat aralıkla 6 veri noktası seç
        selected_indices = [start_index + i * 4 for i in range(6)]

        for idx in selected_indices:
            if idx >= len(hourly_forecast):
                break

            current = hourly_forecast[idx]

            forecast_dt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=(idx - start_index))
            hour = forecast_dt.hour
            month = forecast_dt.month

            analysis = self.ml_engine.predict_all(current, hour, month)

            activity_predictions = self.activity_engine.predict_activity_recommendations(
                weather=current,
                dt=forecast_dt
            )

            top_activities = self.activity_engine.get_top_activities(
                weather=current,
                dt=forecast_dt,
                top_k=3
            )

            prompt = self.prompt_builder.build_prompt(
                weather=current,
                analysis=analysis,
                user_request=effective_prefs
            )

            final_message = self.llm_engine.generate_text(prompt)

            results.append({
                "datetime": forecast_dt.isoformat(),
                "weather_report": current,
                "analysis_metrics": analysis,
                "recommendation": final_message,
                "activity_predictions": activity_predictions,
                "top_activities": top_activities,
            })

        return {
            "status": "success",
            "city": city_name,
            "latitude": lat,
            "longitude": lon,
            "hourly_forecast_24h_4h_step": results,
        }
        

    def geocode_city(self, city_name: str):
        url = "https://geocoding-api.open-meteo.com/v1/search"

        params = {
            "name": city_name,
            "count": 1,
            "language": "tr",
            "format": "json"
     }

        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()

        data = r.json()
        results = data.get("results", [])

        if not results:
            raise ValueError("Şehir bulunamadı")

        result = results[0]

        return (
            result["latitude"],
            result["longitude"],
            result.get("timezone", "auto")
        )
    
    def _resolve_coordinates(self, request):
        if request.city is not None and request.city.strip() != "" and request.city != "string":
            city_name = request.city.strip()
            lat, lon, _ = self.geocode_city(request.city.strip())
            return city_name, lat, lon

        if request.latitude is not None and request.longitude is not None:
            city_name = self.reverse_geocode(request.latitude, request.longitude)
            return city_name, request.latitude, request.longitude

        raise ValueError("Location information is missing.")
    


    def reverse_geocode(self, latitude: float, longitude: float) -> str:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "jsonv2",
            "accept-language": "tr"
        }
        headers = {
            "User-Agent": "weatherwise/1.0"
        }

        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()

        data = r.json()

        address = data.get("address", {})
        return (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or data.get("name")
            or "Unknown"
    )

    async def get_daily_aggregated_forecast(self, user_request: WeatherRequest):
        city_name, lat, lon = self._resolve_coordinates(user_request)
        weather_full = await self.fetch_weather(lat, lon)
        effective_prefs = self._get_effective_preferences(user_request)
        
        hourly_forecast = weather_full["hourly_forecast"]

        daily_results = self.aggregation_service.aggregate_daily_forecast(
            hourly_forecast=hourly_forecast,
            ml_engine=self.ml_engine
        )

        final_daily_results = []

        for day_result in daily_results:
            prompt = self.prompt_builder.build_daily_prompt(
                daily_weather_summary=day_result["weather_summary"],
                daily_analysis=day_result["analysis_summary"],
                user_request=effective_prefs
            )

            final_message = self.llm_engine.generate_text(prompt)

            final_daily_results.append({
                "date": day_result["date"],
                "weather_summary": day_result["weather_summary"],
                "analysis_summary": day_result["analysis_summary"],
                "daily_recommendation": final_message,
                "hourly_details": day_result["hourly_details"]
            })


        return {
            "status": "success",
            "city": city_name,
            "latitude": lat,
            "longitude": lon,
            "timezone": weather_full["metadata"]["timezone"],
            "daily_forecast": final_daily_results   
    }