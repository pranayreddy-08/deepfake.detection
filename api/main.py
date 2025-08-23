import sys
from pathlib import Path

# ensure imports like "src.*" work when running uvicorn
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, UploadFile, File, Query
from src.models.predict_image import predict_image_bytes

app = FastAPI(title="Deepfake Detection API")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict-image")
async def predict_image(
    image: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0)
):
    data = await image.read()
    out = predict_image_bytes(data, threshold=threshold)  # uses saved Best-J if threshold=None
    return out
