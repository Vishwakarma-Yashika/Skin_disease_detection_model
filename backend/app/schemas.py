from pydantic import BaseModel


class ClassProbability(BaseModel):
    key: str
    full_name: str
    probability: float
    category: str  # "malignant" | "benign" | "unknown"


class PredictionResponse(BaseModel):
    decision: str  # "out_of_scope" | "refer_malignancy" | "refer_uncertain" | "automated"
    decision_title: str
    decision_detail: str
    # The fields below are empty when the decision is "out_of_scope": no diagnosis is given.
    predicted_key: str | None
    predicted_name: str | None
    confidence: float | None
    malignancy_probability: float | None
    category: str | None  # "malignant" | "benign"
    description: str | None
    probabilities: list[ClassProbability]
    gradcam_overlay: str | None
    model_name: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int | None
    model_name: str | None
    detail: str | None = None
