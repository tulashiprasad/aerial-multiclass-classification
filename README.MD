# Aerial Multiclass Classification API

FastAPI service that exposes a YOLOv8 segmentation/classification pipeline for aerial imagery. It accepts an uploaded image, runs inference via Ultralytics, returns structured detections, and stores an annotated overlay for auditing. Training and data-prep scripts are included for fine-tuning on custom datasets.


## Requirements
- Python 3.12
- pip / virtualenv (recommended)
- CUDA-capable GPU optional but recommended for training/inference speed

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .sample.env .env
```
 **Note:**
*If you encounter any issues related to your CUDA version, please install the PyTorch build that matches your local CUDA setup*

## Running the API
```bash
source .venv/bin/activate
python -m uvicorn src.main:app --reload
```
### Health Check
- `GET /health` → `{ status, app_name, version }`

### Perform Inference
```bash 
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

## Training Your Own Model
1. Place dataset under `datasets/` (default structure follows YOLO format with `train/`, `valid/`, `test/` subfolders).
2. Optionally run `python src/scripts/oversample.py` to duplicate minority class samples.
3. Start training:
   ```bash
   python src/train.py
   ```

## Written Response Questions
-  The written Response questions are in `assignment/Research_Engineer_Test.pdf` file