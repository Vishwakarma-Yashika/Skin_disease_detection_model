from pydantic import BaseModel


class ClassProbability(BaseModel):
    key: str
    full_name: str
    probability: float
    category: str  # "malignant" | "benign" | "unknown"


class PredictionResponse(BaseModel):
    predicted_key: str
    predicted_name: str
    confidence: float
    category: str  # "malignant" | "benign"
    risk_level: str
    description: str
    typical_action: str
    probabilities: list[ClassProbability]
    gradcam_overlay: str | None
    model_val_accuracy: float | None
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int | None
    model_val_accuracy: float | None
    detail: str | None = None
