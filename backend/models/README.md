The app now serves the E0 ensemble listed in `research/outputs/decision_system/manifest.json`;
set `ENSEMBLE_MANIFEST` to use another location. The older checkpoint below is not served and is
used only by `research/evaluation/step1_reliability.py`.

Put `best_fusion_model.pt` here.

It's the checkpoint saved by the Colab notebook's training cell, at:

    /content/drive/MyDrive/skin_disease_project_outputs/checkpoints/best_fusion_model.pt

Copy/download that file from Google Drive and place it at:

    backend/models/best_fusion_model.pt

Or point the backend at a different location with an env var:

    export MODEL_PATH=/path/to/best_fusion_model.pt
