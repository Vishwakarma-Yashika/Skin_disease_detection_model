import os

# Path to the trained checkpoint produced by the Colab training notebook
# (Part "1.5 MODEL & TRAINING" -> best_fusion_model.pt). Override with the
# MODEL_PATH env var if you keep it somewhere else.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_fusion_model.pt"),
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
