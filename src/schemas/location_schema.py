from pydantic import BaseModel
from typing import Optional


class CityNameResponse(BaseModel):
    status: str
    city: str
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None