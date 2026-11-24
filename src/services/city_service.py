import requests
from typing import List, Dict, Any


class CityService:
    def __init__(self):
        self.base_url = "https://geocoding-api.open-meteo.com/v1/search"

        # Şehir / kasaba benzeri yerleşimler
        # PPL   = populated place
        # PPLA  = seat of a first-order admin division
        # PPLA2 = seat of a second-order admin division
        # PPLA3 = seat of a third-order admin division
        # PPLA4 = seat of a fourth-order admin division
        # PPLC  = capital
        # PPLG  = seat of government
        # PPLL  = populated locality
        # PPLS  = populated places
        # PPT   = populated place? (bazı veri varyasyonlarında gelebilir)
        self.allowed_feature_codes = {
            "PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4",
            "PPLC", "PPLG", "PPLL", "PPLS"
        }

    def search_cities(self, query: str, count: int = 10, language: str = "en") -> List[Dict[str, Any]]:
        query = query.strip()

        if len(query) < 2:
            return []

        params = {
            "name": query,
            "count": min(count, 20),   # aşırı yüklenmeyelim
            "language": language,
            "format": "json"
        }

        response = requests.get(self.base_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        raw_results = data.get("results", [])

        filtered_results = []
        seen = set()

        for item in raw_results:
            feature_code = item.get("feature_code", "")
            city_name = item.get("name")
            country = item.get("country")
            country_code = item.get("country_code")
            latitude = item.get("latitude")
            longitude = item.get("longitude")

            # Temel zorunlu alanlar yoksa geç
            if not all([city_name, country, country_code, latitude, longitude]):
                continue

            # Sadece şehir benzeri feature_code kabul et
            if feature_code not in self.allowed_feature_codes:
                continue

            # Tekrarları engelle
            key = (city_name.lower(), country_code, latitude, longitude)
            if key in seen:
                continue
            seen.add(key)

            filtered_results.append({
                "city": city_name,
                "country": country,
                "country_code": country_code,
                "latitude": latitude,
                "longitude": longitude
            })

        return filtered_results