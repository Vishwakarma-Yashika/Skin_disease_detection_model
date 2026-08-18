"""
Model architecture + checkpoint loading.

This MUST stay structurally identical to the training cell in the Colab
notebook ("1.5 MODEL & TRAINING") -- it's the same EfficientNet-B0 + ResNet-34
fusion classifier, reproduced here so `best_fusion_model.pt`'s state_dict
loads cleanly.
"""
import logging
import threading

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from app.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH

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


class ModelRegistry:
    """
    Lazily loads the checkpoint once and holds the model + class list in memory.
    Thread-safe singleton so FastAPI's threadpool-executed sync endpoints don't
    race on first load.
    """

    _lock = threading.Lock()
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.diagnosis_classes: list[str] | None = None
        self.val_acc: float | None = None
        self.epoch: int | None = None
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
                logger.info("Loading checkpoint from %s (device=%s)", MODEL_PATH, self.device)
                checkpoint = torch.load(MODEL_PATH, map_location=self.device)
                diagnosis_classes = checkpoint["diagnosis_classes"]
                model = FusionClassifier(num_classes=len(diagnosis_classes), feature_dim=512, pretrained=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()

                self.model = model
                self.diagnosis_classes = diagnosis_classes
                self.val_acc = checkpoint.get("val_acc")
                self.epoch = checkpoint.get("epoch")
                self.loaded = True
                logger.info(
                    "Model loaded: %d classes, val_acc=%.4f (epoch %s)",
                    len(diagnosis_classes),
                    self.val_acc or -1,
                    self.epoch,
                )
            except FileNotFoundError:
                self.load_error = (
                    f"Checkpoint not found at {MODEL_PATH}. Copy best_fusion_model.pt from your "
                    "Drive (skin_disease_project_outputs/checkpoints/) into backend/models/, or set "
                    "the MODEL_PATH env var."
                )
                logger.error(self.load_error)
            except Exception as exc:  # noqa: BLE001 -- surface any load failure to the API
                self.load_error = f"Failed to load checkpoint: {exc!r}"
                logger.exception("Model load failed")

    @property
    def target_layer(self):
        """Last conv block of the EfficientNet branch -- used for Grad-CAM."""
        assert self.model is not None
        return self.model.backbone.eff_features[-1]
