from pydantic import BaseModel
from typing import List


class CitySuggestionItem(BaseModel):
    city: str
    country: str
    country_code: str
    latitude: float
    longitude: float


class CitySuggestionResponse(BaseModel):
    results: List[CitySuggestionItem]