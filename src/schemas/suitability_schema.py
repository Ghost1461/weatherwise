from pydantic import BaseModel

class WeatherInputS(BaseModel):
    temperature_c: float
    feels_like_c: float
    dew_point_c: float
    humidity_pct: float
    pressure_hpa: float
    wind_speed_kmh: float
    wind_direction_deg: float
    wind_gust_kmh: float
    cloud_cover_pct: float
    visibility_km: float
    uv_index: float
    hour_of_day: int
    month: int