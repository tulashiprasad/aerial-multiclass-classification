from fastapi import APIRouter, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from typing import List
from starlette.concurrency import run_in_threadpool

from src.core.logging import logger
from src.handler import ClassificationHandler
from src.core.config import settings

router = APIRouter(tags=["predict"])

classifier = ClassificationHandler()


class PredictionItem(BaseModel):
    label: str          
    confidence: float   
    bbox: List[float] 


class PredictionResponse(BaseModel):
    prediction: List[PredictionItem]


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or file.content_type not in settings.allowed_mimetypes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file must be one of the following types: {', '.join(settings.allowed_mimetypes)}",
        )

    image_bytes = await file.read()

    try:
        classification_output = await run_in_threadpool(
            classifier.classify, image_bytes
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )

    predictions_payload = classification_output.get("predictions", [])
    if not isinstance(predictions_payload, list):
        logger.error("Classifier returned invalid predictions payload")
        raise HTTPException(
            status_code=500,
            detail="Prediction output is invalid",
        )

    items = [PredictionItem(**p) for p in predictions_payload]

    return PredictionResponse(prediction=items)
