# Deepfake Video Detection (Image + Video)

> **Status:** MVP ✅ — trainable model + Streamlit UIs + FastAPI API  
> **Backbone:** Xception (ImageNet) • **TF** 2.15 • **OS:** Windows-friendly (Git Bash / VS Code)

This repo detects deepfakes from **face images** and **videos**.  
You can:
- Train on a small face dataset (e.g., CIPLab real/fake faces).
- Evaluate on a validation split (AUC, confusion, best threshold).
- Run a **Streamlit UI** (image & video) and a **FastAPI** inference API.
