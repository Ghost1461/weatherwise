from pydantic import BaseModel

class ActivityInput(BaseModel):
    month: int
    is_weekend: int
    season: str
    weather_condition: str
    temperature_c: float
    feels_like_c: float
    humidity_pct: float
    wind_speed_kmh: float
    precipitation_mm: float
    cloud_cover_pct: float
    uv_index: float
    hour_of_day: int