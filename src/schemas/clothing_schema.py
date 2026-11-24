from pydantic import BaseModel

class WeatherInputC(BaseModel):
    temperature_c: float
    wind_speed_kmh: float
    wind_gust_kmh: float
    uv_index: float
    hour_of_day: int
    month: int