# Skin Lesion Classifier

Full-stack app around the EfficientNet-B0 + ResNet-34 fusion model trained on
HAM10000 in the companion Colab notebook (`Untitled1.ipynb`). Upload a lesion
photo, get the predicted diagnosis class, a confidence breakdown across all 7
classes, and a Grad-CAM heatmap showing what the model focused on.

**This is a research/demo tool, not a medical device.** Every prediction ships
with a disclaimer in the UI -- keep it there if you extend this.

## Structure

```
backend/    FastAPI service that loads the checkpoint and serves predictions
frontend/   Next.js (App Router + TypeScript + Tailwind) upload/results UI
```

## 1. Get the trained checkpoint

The training notebook saves `best_fusion_model.pt` to your Google Drive at:

    /content/drive/MyDrive/skin_disease_project_outputs/checkpoints/best_fusion_model.pt

Download it from Drive and place it at:

    backend/models/best_fusion_model.pt

(~100 MB, val accuracy 83.5% -- see backend/models/README.md if you'd rather
point at a different path via the `MODEL_PATH` env var.)

## 2. Run the backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Check it loaded the model: `curl http://localhost:8000/api/health` should
report `"model_loaded": true`. If you don't have an NVIDIA GPU locally it'll
just run on CPU -- a single prediction still only takes a second or two.

## 3. Run the frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open http://localhost:3000, upload a lesion photo, click "Analyze image."

## What you get per prediction

- Top predicted class (of the 7 HAM10000 classes: akiec, bcc, bkl, df, mel, nv, vasc) with confidence
- A plain-language description and "what to do" note per class
- A malignant/benign + risk-level badge
- Full probability breakdown across all 7 classes, ranked
- A Grad-CAM heatmap overlay toggle -- shows the image regions that drove the prediction
- The model's own validation accuracy (83.5%), shown alongside every result for context
- A persistent, non-dismissable disclaimer

## Known limitations (carried over from the training notebook)

- Trained on HAM10000 only -- ISIC/DermNet/Fitzpatrick17k weren't available yet at training time (see the notebook's own "DATASET SUFFICIENCY FLAGS" cell).
- No skin-tone debiasing has been applied to this checkpoint (Part 3's `DebiasedFusionModel` in the notebook is the documented next step).
- Val accuracy is 83.5% overall, but per-class performance varies a lot -- e.g. `mel` (melanoma) precision is only 0.56, meaning over 40% of its melanoma flags are false positives. Never treat this as a diagnosis.

## Extending this

- Swap in a debiased checkpoint once you've run Part 3's adversarial debiasing -- no frontend changes needed, the API contract stays the same.
- To deploy, containerize each service (a `Dockerfile` per folder) and set `CORS_ORIGINS` (backend) / `NEXT_PUBLIC_API_URL` (frontend) to your real domains.
