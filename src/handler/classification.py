
from pathlib import Path
from typing import List, Dict, Any
import io
from uuid import uuid4

from PIL import Image
from ultralytics import YOLO
from src.core.config import settings


class ClassificationHandler:
    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            model_path = Path(settings.model_path)

        self.model = YOLO(str(model_path))

    def classify(self, image_bytes: bytes) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self.model(img)[0]

        predictions: List[Dict[str, Any]] = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist() 
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names.get(class_id, str(class_id))

                predictions.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        annotated_bgr = results.plot()          # (H, W, 3) BGR
        annotated_rgb = annotated_bgr[..., ::-1]  # convert BGR -> RGB

        annotated_img = Image.fromarray(annotated_rgb)
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{uuid4().hex}.png"
        annotated_img.save(output_path)

        return {
            "predictions": predictions,
            "overlay_path": str(output_path),
        }
