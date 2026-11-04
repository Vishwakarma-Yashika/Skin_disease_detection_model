"""
Model architecture + checkpoint loading.

This MUST stay structurally identical to the training cell in the Colab
notebook ("1.5 MODEL & TRAINING") -- it's the same EfficientNet-B0 + ResNet-34
fusion classifier, reproduced here so `best_fusion_model.pt`'s state_dict
loads cleanly.
"""
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from app.config import ENSEMBLE_MANIFEST, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, REPO_ROOT

logger = logging.getLogger("skin_app.model")


class EfficientNetResNetFusion(nn.Module):
    """Concat-fusion backbone: EfficientNet-B0 pooled features + ResNet-34 pooled features."""

    def __init__(self, feature_dim=512, pretrained=False, dropout=0.3):
        super().__init__()
        weights_eff = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        weights_res = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None

        effnet = torchvision.models.efficientnet_b0(weights=weights_eff)
        self.eff_features = effnet.features
        self.eff_pool = nn.AdaptiveAvgPool2d(1)
        eff_out_dim = 1280

        resnet = torchvision.models.resnet34(weights=weights_res)
        self.res_features = nn.Sequential(*list(resnet.children())[:-2])
        self.res_pool = nn.AdaptiveAvgPool2d(1)
        res_out_dim = 512

        self.fusion = nn.Sequential(
            nn.Linear(eff_out_dim + res_out_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        ef = self.eff_pool(self.eff_features(x)).flatten(1)
        rf = self.res_pool(self.res_features(x)).flatten(1)
        fused = torch.cat([ef, rf], dim=1)
        return self.fusion(fused)


class FusionClassifier(nn.Module):
    """backbone (feature extractor) + linear classification head."""

    def __init__(self, num_classes, feature_dim=512, pretrained=False):
        super().__init__()
        self.backbone = EfficientNetResNetFusion(feature_dim=feature_dim, pretrained=pretrained)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# Inference-time transform -- identical to `val_transform` in training (no augmentation)
inference_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


@dataclass
class EnsemblePrediction:
    probs: list[float]  # mean of the members' temperature-scaled probabilities, in class order
    malignancy_probability: float
    uncertainty: float  # 1 - top probability
    out_of_scope_score: float
    decision: str  # out_of_scope | refer_malignancy | refer_uncertain | automated


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    """
    Lazily loads the ensemble once and holds its members + class list in memory.
    Thread-safe singleton so FastAPI's threadpool-executed sync endpoints don't
    race on first load.
    """

    _lock = threading.Lock()
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members: list[nn.Module] = []
        self.temperatures: list[float] = []
        self.feature_bank: list[torch.Tensor] = []
        self.thresholds: dict | None = None
        self.neighbours: int | None = None
        self.malignant_idx: list[int] = []
        self.name: str | None = None
        self.diagnosis_classes: list[str] | None = None
        self.loaded = False
        self.load_error: str | None = None

    @classmethod
    def get(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ModelRegistry()
        return cls._instance

    def ensure_loaded(self):
        if self.loaded or self.load_error:
            return
        with self._lock:
            if self.loaded or self.load_error:
                return
            try:
                logger.info("Loading ensemble from %s (device=%s)", ENSEMBLE_MANIFEST, self.device)
                with open(ENSEMBLE_MANIFEST, encoding="utf-8") as f:
                    manifest = json.load(f)
                classes = manifest["classes"]
                members, temperatures = [], []
                for member in manifest["members"]:
                    path = os.path.join(REPO_ROOT, member["checkpoint"])
                    if _sha256(path) != member["sha256"]:
                        raise ValueError(f"{path} does not match the checkpoint hash in the manifest")
                    checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                    if list(checkpoint["diagnosis_classes"]) != classes:
                        raise ValueError(f"{path} has a different class order from the manifest")
                    model = FusionClassifier(num_classes=len(classes), feature_dim=512, pretrained=False)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(self.device)
                    model.eval()
                    members.append(model)
                    temperatures.append(float(member["temperature"]))
                with np.load(os.path.join(REPO_ROOT, manifest["feature_bank"])) as bank:
                    features = bank["features"]
                if len(features) != len(members):
                    raise ValueError("The feature bank does not match the ensemble members")

                self.members = members
                self.temperatures = temperatures
                self.feature_bank = [torch.from_numpy(f).to(self.device) for f in features]
                self.thresholds = manifest["thresholds"]
                self.neighbours = manifest["out_of_scope"]["k"]
                self.malignant_idx = [classes.index(c) for c in manifest["malignant_classes"]]
                self.name = manifest["name"]
                self.diagnosis_classes = classes
                self.loaded = True
                logger.info("Ensemble loaded: %d members, %d classes", len(members), len(classes))
            except FileNotFoundError as exc:
                self.load_error = (
                    f"Ensemble file not found: {exc.filename}. Run research.evaluation.decision_rule_eval, "
                    "or set the ENSEMBLE_MANIFEST env var."
                )
                logger.error(self.load_error)
            except Exception as exc:  # noqa: BLE001 -- surface any load failure to the API
                self.load_error = f"Failed to load the ensemble: {exc!r}"
                logger.exception("Ensemble load failed")

    @torch.no_grad()
    def predict(self, input_tensor) -> EnsemblePrediction:
        """Applies the decision rule from research/configs/decision_rule.yaml to one image."""
        member_probs, distances = [], []
        for model, temperature, bank in zip(self.members, self.temperatures, self.feature_bank):
            features = model.backbone(input_tensor)
            member_probs.append(torch.softmax(model.classifier(features).double() / temperature, dim=1))
            similarity = torch.nn.functional.normalize(features, dim=1) @ bank.T
            distances.append(1.0 - similarity.topk(self.neighbours, dim=1).values.mean().item())
        probs = torch.stack(member_probs).mean(dim=0)[0]
        malignancy = probs[self.malignant_idx].sum().item()
        uncertainty = 1.0 - probs.max().item()
        out_of_scope_score = float(np.mean(distances))

        if out_of_scope_score > self.thresholds["out_of_scope"]:
            decision = "out_of_scope"
        elif malignancy >= self.thresholds["p_mal"]:
            decision = "refer_malignancy"
        elif uncertainty > self.thresholds["msp"]:
            decision = "refer_uncertain"
        else:
            decision = "automated"
        return EnsemblePrediction(probs.cpu().tolist(), malignancy, uncertainty, out_of_scope_score, decision)

    @property
    def gradcam_model(self):
        """Grad-CAM explains one member, the first in the manifest."""
        assert self.members
        return self.members[0]

    @property
    def target_layer(self):
        """Last conv block of the EfficientNet branch -- used for Grad-CAM."""
        return self.gradcam_model.backbone.eff_features[-1]
