import os

# Path to the trained checkpoint produced by the Colab training notebook
# (Part "1.5 MODEL & TRAINING" -> best_fusion_model.pt). Override with the
# MODEL_PATH env var if you keep it somewhere else.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_fusion_model.pt"),
)

# The served system: the E0 ensemble, its temperatures, thresholds and out-of-scope reference.
# Written by research.evaluation.decision_rule_eval. Paths inside it are relative to the repo.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENSEMBLE_MANIFEST = os.environ.get(
    "ENSEMBLE_MANIFEST",
    os.path.join(REPO_ROOT, "research", "outputs", "decision_system", "manifest.json"),
)

# Must match training: torchvision.transforms Resize((224, 224)) + ImageNet norm
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Origins allowed to call this API. Add your deployed frontend URL here later.
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

# Max upload size (bytes) -- 10 MB is generous for a dermoscopy photo
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
