import io
import logging

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.class_info import DISCLAIMER, get_class_info
from app.config import CORS_ORIGINS, MAX_UPLOAD_BYTES
from app.gradcam import gradcam_overlay_base64
from app.model import ModelRegistry, inference_transform
from app.schemas import ClassProbability, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin_app")

app = FastAPI(
    title="Skin Lesion Classifier API",
    description=(
        "Serves the EfficientNet-B0 + ResNet-34 fusion model trained on HAM10000 "
        "in the companion Colab notebook. Research/demo use only -- see /health "
        "and each prediction's `disclaimer` field."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model_on_startup():
    # Load eagerly so the first real request isn't slow, but don't crash the
    # server if the checkpoint is missing -- /health and /predict will report why.
    ModelRegistry.get().ensure_loaded()


@app.get("/api/health", response_model=HealthResponse)
def health():
    registry = ModelRegistry.get()
    registry.ensure_loaded()
    return HealthResponse(
        status="ok" if registry.loaded else "model_unavailable",
        model_loaded=registry.loaded,
        device=str(registry.device),
        num_classes=len(registry.diagnosis_classes) if registry.diagnosis_classes else None,
        model_val_accuracy=registry.val_acc,
        detail=registry.load_error,
    )


@app.get("/api/classes")
def list_classes():
    """Static reference info for every class the model can predict -- lets the
    frontend render an info panel without needing a prediction first."""
    registry = ModelRegistry.get()
    registry.ensure_loaded()
    keys = registry.diagnosis_classes or ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    return {key: {**get_class_info(key), "key": key} for key in keys}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    registry = ModelRegistry.get()
    registry.ensure_loaded()
    if not registry.loaded:
        raise HTTPException(
            status_code=503,
            detail=registry.load_error or "Model is not loaded yet.",
        )

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image file, got '{file.content_type}'.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB).")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not read that file as an image.") from exc
    image = image.convert("RGB")

    input_tensor = inference_transform(image).unsqueeze(0).to(registry.device)

    with torch.no_grad():
        logits = registry.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()

    ranked = sorted(zip(registry.diagnosis_classes, probs), key=lambda kv: kv[1], reverse=True)
    top_key, top_prob = ranked[0]
    top_info = get_class_info(top_key)

    probabilities = [
        ClassProbability(
            key=key,
            full_name=get_class_info(key)["full_name"],
            probability=round(prob, 6),
            category=get_class_info(key)["category"],
        )
        for key, prob in ranked
    ]

    top_idx = registry.diagnosis_classes.index(top_key)
    try:
        gradcam_data_url = gradcam_overlay_base64(
            registry.model, registry.target_layer, input_tensor, image, class_idx=top_idx
        )
    except Exception:  # noqa: BLE001 -- Grad-CAM is a nice-to-have, never block the prediction on it
        logger.exception("Grad-CAM failed; returning prediction without an overlay")
        gradcam_data_url = None

    return PredictionResponse(
        predicted_key=top_key,
        predicted_name=top_info["full_name"],
        confidence=round(top_prob, 6),
        category=top_info["category"],
        risk_level=top_info["risk_level"],
        description=top_info["description"],
        typical_action=top_info["typical_action"],
        probabilities=probabilities,
        gradcam_overlay=gradcam_data_url,
        model_val_accuracy=registry.val_acc,
        disclaimer=DISCLAIMER,
    )
