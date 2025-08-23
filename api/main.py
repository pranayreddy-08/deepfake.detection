

import sys
from pathlib import Path

# ensure "src.*" imports work when running uvicorn from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, UploadFile, File, Query
from tempfile import NamedTemporaryFile
from src.models.predict_image import predict_image_bytes
from src.models.predict_video import predict_video

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

@app.post("/predict-video")
async def predict_video_api(
    video: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    every_n: int = Query(default=25, ge=1, description="Sample every Nth frame."),
    max_faces: int = Query(default=128, ge=1, description="Max faces to score across the video.")
):
    # Persist to a temp file so OpenCV can read it
    with NamedTemporaryFile(delete=True, suffix=f"_{video.filename}") as tmp:
        tmp.write(await video.read())
        tmp.flush()
        out = predict_video(Path(tmp.name), threshold=threshold, every_n=every_n, max_faces=max_faces)
    return out
