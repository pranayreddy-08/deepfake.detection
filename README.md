# Deepfake Video Detection
├─ api/ # FastAPI service (inference API)
│ └─ main.py
├─ app/ # Streamlit demo UI
│ └─ app.py
├─ src/ # Library code (preprocess, train, infer)
│ ├─ data/
│ ├─ models/
│ ├─ config.py
│ └─ init.py
├─ data/ # Local data (kept out of git except .gitkeep)
│ ├─ raw/
│ ├─ frames/
│ └─ faces/
├─ saved_models/ # Trained weights (ignored from git)
├─ requirements.txt
└─ README.md
