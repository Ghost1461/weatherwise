#Endpoint’ler burada olur(controller gibi)

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from src.schemas.umbrella_schema import WeatherInput
from src.schemas.clothing_schema import WeatherInputC
from src.schemas.suitability_schema import WeatherInputS
from src.schemas.schema_24h import Weather24hInput
from src.schemas.activity_type_schema import ActivityInput
from src.schemas.activity_type_schema import ActivityInput
from src.schemas.city_schema import CitySuggestionResponse

from src.services.prediction_service import predict_umbrella_service
from src.services.prediction_service import predict_suitability_service
from src.services.prediction_service import predict_clothing_service
from src.services.prediction_service import predict_all_service
from src.services.prediction_service import predict_activity_service
from src.services.prediction_service import predict_activity_service
from src.services.city_service import CityService
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import HTTPException
from src.schemas.open_meteo_schema import WeatherRequest, OpenMeteoParams
from src.services.advice_services import AdviceService

router = APIRouter()

city_service = CityService()


@router.get("/")
def home():
    return {"message": "Weatherwise API is running"}

@router.post("/predict/umbrella")
def umbrella_endpoint(data: WeatherInput):
    features = data.model_dump()
    return predict_umbrella_service(features)

@router.post("/predict/suitability")
def suitability_endpoint(data: WeatherInputS):
    features = data.model_dump()
    return predict_suitability_service(features)

@router.post("/predict/clothing")
def clothing_endpoint(data: WeatherInputC):
    features = data.model_dump()
    return predict_clothing_service(features)

@router.post("/predict/all")
def predict_all_endpoint(data: Weather24hInput):
    features = data.model_dump()
    return predict_all_service(features)

@router.post("/predict/activities")
def predict_activities_route(data: ActivityInput):
    try:
        features = data.model_dump()
        result = predict_activity_service(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Sadece AdviceService'i başlatıyoruz. 
# O zaten arkada Weather, ML ve LLM servislerini kendisi yönetiyor.
advice_service = AdviceService()

# --- 1. SADECE HAVA DURUMU ENDPOINT'I ---
# Bu endpoint sadece koordinat bekler, Swagger'da tertemiz görünür.
@router.post("/weather-only", response_model=dict)
async def get_just_weather(request: OpenMeteoParams):
    print("REQUEST BODY:", request.model_dump())
    try:
        city_name, lat, lon = advice_service._resolve_coordinates(request)
        weather = await advice_service.fetch_weather(lat, lon)
        return {
                "status": "success",
                "city": city_name,
                "latitude": lat,
                "longitude": lon,
                **weather
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# --- TAM ÖNERİ ---
# Weather + ML + LLM + activity önerileri
@router.post("/fetch-advice", response_model=dict)
async def fetch_advice_endpoint(request: WeatherRequest):
    try:
        return await advice_service.get_final_recommendation(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- SADECE ACTIVITY ÖNERİLERİ ---
# LLM kullanmadan sadece OpenMeteo + activity engine
@router.post("/activity-advice", response_model=dict)
async def activity_advice_endpoint(request: OpenMeteoParams):
    try:
        city_name, lat, lon = advice_service._resolve_coordinates(request)

        weather_full = await advice_service.fetch_weather(lat, lon)

        hourly_forecast = weather_full["hourly_forecast"]

        tz_name = weather_full["metadata"]["timezone"]
        now = datetime.now(ZoneInfo(tz_name))

        future_hours = []
        for hour_weather in hourly_forecast:
            forecast_dt = datetime.fromisoformat(hour_weather["datetime"])
            if forecast_dt >= now:
                future_hours.append(hour_weather)

        next_24h = future_hours[:24]

        hourly_activity_results = []

        for hour_weather in next_24h:
            forecast_dt = datetime.fromisoformat(hour_weather["datetime"])

            activity_predictions = advice_service.activity_engine.predict_activity_recommendations(
                weather=hour_weather,
                dt=forecast_dt
            )

            top_activities = advice_service.activity_engine.get_top_activities(
                weather=hour_weather,
                dt=forecast_dt,
                top_k=3
            )

            suggested_activities = [item["activity"] for item in top_activities]

            hourly_activity_results.append({
                "datetime": forecast_dt.isoformat(),
                "weather_report": hour_weather,
                "activity_predictions": activity_predictions,
                "top_activities": top_activities,
                "suggested_activities": suggested_activities
            })

        return {
            "status": "success",
            "city": city_name,
            "latitude": lat,
            "longitude": lon,
            "timezone": tz_name,
            "activity_advice_24h": hourly_activity_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/cities/search", response_model=CitySuggestionResponse)
def search_cities(
    q: str = Query(..., min_length=2),
    count: int = Query(10, ge=1, le=20),
    language: str = Query("en")
):
    try:
        results = city_service.search_cities(query=q, count=count, language=language)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"City search failed: {str(e)}")
    
@router.post("/daily-summary")
async def get_daily_summary_advice(request: WeatherRequest):
    try:
        return await advice_service.get_daily_aggregated_forecast(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    