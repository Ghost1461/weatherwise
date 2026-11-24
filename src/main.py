from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Weatherwise API")

app.include_router(router)

