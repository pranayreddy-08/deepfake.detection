
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.models.predict_video import predict_video

st.set_page_config(page_title="Deepfake Face Detector — Video", layout="centered")
st.title("Deepfake Face Detector — Video")

st.caption(
    "Upload a short MP4 with a visible face. The app samples frames, detects faces, "
    "scores each face with the trained model, and averages the scores."
)

col1, col2 = st.columns(2)
every_n = col1.number_input("Sample every Nth frame", min_value=1, max_value=60, value=25, step=1)
max_faces = col2.number_input("Max faces to score", min_value=16, max_value=1000, value=128, step=16)

col3, col4 = st.columns(2)
use_bestj = col3.radio("Threshold mode", ["Best-J (from eval)", "0.5 (Recall-heavy)"], index=0)
manual = col4.checkbox("Manual threshold")
thr_manual = None
if manual:
    thr_manual = col4.slider("Manual thr", 0.0, 1.0, 0.75, 0.01)

video_file = st.file_uploader("Upload .mp4", type=["mp4", "mov", "mkv"])
tmp_path = Path("tmp_video.mp4")

if video_file:
    tmp_path.write_bytes(video_file.read())
    st.video(str(tmp_path))

    with st.spinner("Analyzing video..."):
        thr = None
        if thr_manual is not None:
            thr = float(thr_manual)
        elif use_bestj.endswith("(Recall-heavy)"):
            thr = 0.5  # explicit recall-heavy
        # else thr=None -> use saved Best-J
        out = predict_video(tmp_path, threshold=thr, every_n=int(every_n), max_faces=int(max_faces))

    if out["faces_scored"] == 0:
        st.error("No faces detected. Try a video with a clear frontal face, or reduce 'every_n'.")
    else:
        st.success(f"Prediction: **{out['pred_label']}**")
        st.write(
            f"Frames scanned: `{out['frames_scanned']}` • Faces scored: `{out['faces_scored']}`  \n"
            f"Mean prob REAL: `{out['prob_real_mean']:.3f}` • Median: `{out['prob_real_median']:.3f}`  \n"
            f"Threshold used: `{out['threshold']:.3f}`"
        )
        st.caption(f"Sample of first scores: {out['all_probs_sample']}")

