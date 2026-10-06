"""
E0 model = the backend's FusionClassifier, imported rather than copied, so the
research training code and the serving app share a single architecture
definition (and E0 checkpoints load in the backend unchanged).

backend/app/ is not modified; it is only put on sys.path.
"""
import sys
from pathlib import Path

import torch.nn as nn
from torchvision.ops import StochasticDepth

_BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.model import FusionClassifier  # noqa: E402
from app.model import inference_transform as backend_inference_transform  # noqa: E402

ARCHITECTURE_ID = "efficientnet_b0+resnet34_concat_fusion"
# backend/app/model.py hard-codes these torchvision weight enums when pretrained=True.
BACKEND_PRETRAINED_WEIGHTS = {"efficientnet_b0": "IMAGENET1K_V1", "resnet34": "IMAGENET1K_V1"}


def build_model(cfg: dict, pretrained: bool) -> FusionClassifier:
    m = cfg["model"]
    if m["architecture"] != ARCHITECTURE_ID:
        raise ValueError(f"E0 requires architecture {ARCHITECTURE_ID!r}, config says {m['architecture']!r}")
    if pretrained and m["pretrained_weights"] != BACKEND_PRETRAINED_WEIGHTS:
        raise ValueError(f"Config pretrained_weights {m['pretrained_weights']} != backend {BACKEND_PRETRAINED_WEIGHTS}")
    if m["num_classes"] != len(cfg["data"]["classes"]):
        raise ValueError("model.num_classes does not match data.classes")

    model = FusionClassifier(num_classes=m["num_classes"], feature_dim=m["fusion_dim"], pretrained=pretrained)
    verify_architecture(model, cfg)
    return model


def verify_architecture(model: FusionClassifier, cfg: dict) -> dict:
    """Fail loudly if the instantiated network differs from the declared E0 architecture."""
    m = cfg["model"]
    fusion = model.backbone.fusion
    expected_types = [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout]
    actual_types = [type(layer) for layer in fusion]
    if actual_types != expected_types:
        raise RuntimeError(f"Fusion head is {actual_types}, expected {expected_types}")

    in_dim = m["efficientnet_feature_dim"] + m["resnet_feature_dim"]
    checks = {
        "fusion_linear_in": (fusion[0].in_features, in_dim),
        "fusion_linear_out": (fusion[0].out_features, m["fusion_dim"]),
        "fusion_bn_features": (fusion[1].num_features, m["fusion_dim"]),
        "dropout_p": (fusion[3].p, m["dropout"]),
        "classifier_in": (model.classifier.in_features, m["fusion_dim"]),
        "classifier_out": (model.classifier.out_features, m["num_classes"]),
        "resnet_last_channels": (model.backbone.res_features[-1][-1].conv2.out_channels, m["resnet_feature_dim"]),
        "efficientnet_last_channels": (model.backbone.eff_features[-1][0].out_channels, m["efficientnet_feature_dim"]),
    }
    mismatched = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatched:
        raise RuntimeError(f"Architecture mismatch (actual, expected): {mismatched}")

    # torchvision scales stochastic depth linearly per block up to (n-1)/n * prob.
    sd = [mod.p for mod in model.backbone.eff_features.modules() if isinstance(mod, StochasticDepth)]
    if not sd or max(sd) > m["efficientnet_stochastic_depth_prob"] or max(sd) <= 0:
        raise RuntimeError(f"Unexpected EfficientNet stochastic depth probabilities: {sd}")
    return {"checks": {k: v[0] for k, v in checks.items()}, "efficientnet_stochastic_depth_max": max(sd)}


def assert_eval_transform_matches_backend(eval_transform) -> None:
    """Validation/test preprocessing must be exactly what the deployed app applies."""
    if repr(eval_transform) != repr(backend_inference_transform):
        raise RuntimeError(
            "Research eval transform differs from backend inference_transform:\n"
            f"research: {eval_transform!r}\nbackend:  {backend_inference_transform!r}"
        )
