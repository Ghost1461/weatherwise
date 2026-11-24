import openmeteo_requests
import requests_cache
from retry_requests import retry
from src.core.config import settings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


class WeatherService:
    def __init__(self):
        # Cache ve Retry kurulumu
        self.cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)
        self.url = settings.OPEN_METEO_URL

    def fetch_raw_weather(self, lat: float, lon: float):
        """Sadece Open-Meteo SDK'sını kullanarak ham veriyi çeker."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dewpoint_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "surface_pressure",
                "cloud_cover",
                "visibility",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m"
            ],
            "forecast_days": 7,
            "timezone": "auto"
        }

        responses = self.openmeteo.weather_api(self.url, params=params)
        return responses[0]

    async def get_formatted_weather(self, lat: float, lon: float):
        """Ham veriyi alıp 168 saatlik TAM parametre setini döndürür."""
        response = self.fetch_raw_weather(lat, lon)
        hourly = response.Hourly()

        # Verileri çek
        temp_c = hourly.Variables(0).ValuesAsNumpy()
        hum_pct = hourly.Variables(1).ValuesAsNumpy()
        dew_c = hourly.Variables(2).ValuesAsNumpy()
        app_temp_c = hourly.Variables(3).ValuesAsNumpy()
        precip_mm = hourly.Variables(4).ValuesAsNumpy()
        w_code = hourly.Variables(5).ValuesAsNumpy()
        press_hpa = hourly.Variables(6).ValuesAsNumpy()
        cloud_pct = hourly.Variables(7).ValuesAsNumpy()
        vis_km = hourly.Variables(8).ValuesAsNumpy()
        w_speed_kmh = hourly.Variables(9).ValuesAsNumpy()
        w_dir_deg = hourly.Variables(10).ValuesAsNumpy()
        w_gust_kmh = hourly.Variables(11).ValuesAsNumpy()

        # Zaman bilgisi
        start_ts = hourly.Time()
        interval = hourly.Interval()
        tz_name = response.Timezone()

        if isinstance(tz_name, bytes):
            tz_name = tz_name.decode("utf-8")

        if not tz_name:
            tz_name = "UTC"

        start_datetime = datetime.fromtimestamp(
            start_ts, tz=timezone.utc
        ).astimezone(ZoneInfo(tz_name))

        forecast_168h = []
        total_hours = len(temp_c)

        for i in range(total_hours):
            dt = start_datetime + timedelta(seconds=i * interval)

            hour_data = {
                "hour_index": i,
                "datetime": dt.isoformat(),
                "date": dt.date().isoformat(),
                "hour": dt.hour,
                "temperature_c": float(temp_c[i]),
                "humidity_pct": float(hum_pct[i]),
                "dew_point_c": float(dew_c[i]),
                "feels_like_c": float(app_temp_c[i]),
                "precipitation_mm": float(precip_mm[i]),
                "weather_condition_code": int(w_code[i]),
                "pressure_hpa": float(press_hpa[i]),
                "cloud_cover_pct": float(cloud_pct[i]),
                "visibility_km": float(vis_km[i]),
                "wind_speed_kmh": float(w_speed_kmh[i]),
                "wind_direction_deg": float(w_dir_deg[i]),
                "wind_gust_kmh": float(w_gust_kmh[i])
            }

            forecast_168h.append(hour_data)

        return {
            "status": "success",
            "metadata": {
                "latitude": response.Latitude(),
                "longitude": response.Longitude(),
                "elevation": response.Elevation(),
                "timezone": tz_name
            },
            "hourly_forecast": forecast_168h
        }