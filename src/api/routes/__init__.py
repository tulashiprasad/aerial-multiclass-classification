"""API routes initialization."""
from fastapi import APIRouter

from src.api.routes import classifier

api_router = APIRouter()

api_router.include_router(classifier.router)
