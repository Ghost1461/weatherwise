from pydantic import BaseModel,Field,model_validator
from typing import Optional

class WeatherRequest(BaseModel):
    latitude: Optional [float] =Field(...,ge=-90, 
        le=90, 
        description="Dünya üzerindeki enlem değeri (-90 ile 90 arası)")
    longitude: Optional [float] = Field(...,ge=-180, 
        le=180, 
        description="Dünya üzerindeki boylam değeri (-180 ile 180 arası)")
    city: Optional[str] = None

    
    # Kişiselleştirme Modu (Genel On/Off)
    personalization_on: bool = False 
    
    # Üçlü Durum (True: On, False: Off, None: Belirtilmemiş/Orta)
    has_dog: Optional[bool] = None
    has_car: Optional[bool] = None
    near_sea: Optional[bool] = None
    sun_allergy: Optional[bool] = None
    tone: Optional[str] = "auto"

    @model_validator(mode="after")
    def validate_location(self):
        has_coords = self.latitude is not None and self.longitude is not None
        has_city = self.city is not None and self.city.strip() != ""

        if not has_coords and not has_city:
            raise ValueError("Either city or both latitude and longitude must be provided.")

        return self


    # Mevcut WeatherRequest'in altına ekle
class OpenMeteoParams(BaseModel):
    latitude: Optional[float] = Field(default=None)
    longitude: Optional[float] = Field(default=None)
    city: Optional[str] = None
    @model_validator(mode="after")
    def validate_location(self):
        has_real_city = (
            self.city is not None and
            self.city.strip() != "" and
            self.city.strip().lower() != "string"
        )

        has_coords = self.latitude is not None and self.longitude is not None

        if not has_coords and not has_real_city:
            raise ValueError("Either city or both latitude and longitude must be provided.")

        return self