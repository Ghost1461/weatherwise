from pydantic import BaseModel, Field
from typing import Optional


class WeatherAdviceInput(BaseModel):
    temperature_c: float = Field(..., example=18.0)
    feels_like_c: float = Field(..., example=17.0)
    dew_point_c: float = Field(..., example=12.0)
    humidity_pct: float = Field(..., example=70.0)
    pressure_hpa: float = Field(..., example=1012.0)
    wind_speed_kmh: float = Field(..., example=15.0)
    wind_direction_deg: float = Field(..., example=180.0)
    wind_gust_kmh: float = Field(..., example=22.0)
    cloud_cover_pct: float = Field(..., example=90.0)
    visibility_km: float = Field(..., example=8.0)
    uv_index: float = Field(..., example=2.0)
    hour_of_day: int = Field(..., example=14)
    month: int = Field(..., example=4)
    season: str = Field(..., example="spring")

    hasPet: Optional[bool] = Field(None, example=True)
    coldSensitive: Optional[bool] = Field(None, example=True)
    sunSensitive: Optional[bool] = Field(None, example=False)
    preferOutdoor: Optional[bool] = Field(None, example=False)
    wantsMovement: Optional[bool] = Field(None, example=True)