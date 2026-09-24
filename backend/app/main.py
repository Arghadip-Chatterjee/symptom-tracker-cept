import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.db import init_db
from app.routers import admin, predict

settings = get_settings()
init_db()

app = FastAPI(
    title="Symptom Tracker RAG API",
    description="Symptom → disease/medicine predictions grounded in PDF books and Kaggle datasets.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(admin.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": settings.collection_name,
        "openai_configured": settings.openai_configured,
    }
