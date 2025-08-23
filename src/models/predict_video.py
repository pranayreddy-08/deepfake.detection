
from pathlib import Path
import numpy as np, cv2, tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from src.config import IMG_SIZE, MODEL_PATH, FRAME_EVERY_N, MIN_FACE, MODEL_DIR

# Load model once
_MODEL = tf.keras.models.load_model(str(MODEL_PATH))
# Face detector (OpenCV Haar)
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _load_best_threshold(default=0.5):
    f = MODEL_DIR / "best_threshold.txt"
    if f.exists():
        try:
            return float(f.read_text().strip())
        except:
            return default
    return default

def _prep_face(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE).astype("float32")
    rgb = preprocess_input(rgb)  # [-1, 1] scaling
    return np.expand_dims(rgb, 0)

def score_video(video_path: Path, every_n: int = FRAME_EVERY_N, max_faces: int = 128):
    """
    Sample every Nth frame, detect faces, run model on each face.
    Returns dict with mean/median prob_real and counts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    probs = []
    frames_scanned = 0
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % max(1, every_n) == 0:
            frames_scanned += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(MIN_FACE, MIN_FACE))

            # If no face found, optionally fallback to a central crop (disabled by default)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                x_in = _prep_face(face)
                p = float(_MODEL.predict(x_in, verbose=0)[0][0])  # prob of REAL (class 1)
                probs.append(p)
                if len(probs) >= max_faces:
                    break
        if len(probs) >= max_faces:
            break
        i += 1

    cap.release()

    result = {
        "frames_scanned": frames_scanned,
        "faces_scored": len(probs),
        "prob_real_mean": float(np.mean(probs)) if probs else None,
        "prob_real_median": float(np.median(probs)) if probs else None,
        "all_probs_sample": [float(x) for x in probs[:20]],  # small peek to inspect
    }
    return result

def predict_video(video_path: Path, threshold: float | None = None, every_n: int = FRAME_EVERY_N, max_faces: int = 128):
    if threshold is None:
        threshold = _load_best_threshold(0.5)
    stats = score_video(video_path, every_n=every_n, max_faces=max_faces)
    prob = stats["prob_real_mean"] if stats["prob_real_mean"] is not None else 0.0
    label = "REAL" if prob >= threshold else "FAKE"
    stats.update({"threshold": float(threshold), "pred_label": label})
    return stats
