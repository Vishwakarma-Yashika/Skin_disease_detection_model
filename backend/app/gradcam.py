"""
Grad-CAM, reused from Part 4 of the audit notebook, adapted to run against a
single uploaded image and return a base64 PNG overlay for the frontend.
"""
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Standard Grad-CAM via forward/backward hooks on a target conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            output = self.model(input_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward(retain_graph=True)

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam, class_idx, logits.detach()
        finally:
            torch.set_grad_enabled(was_grad_enabled)


def overlay_heatmap(image_rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image_rgb_uint8: HxWx3 uint8. cam: HxW float in [0,1]. Returns an RGB overlay."""
    # Simple jet-like colormap without a matplotlib dependency at request time.
    heatmap = _jet_colormap(cam)
    overlay = (alpha * heatmap + (1 - alpha) * image_rgb_uint8).astype(np.uint8)
    return overlay


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """x: HxW float in [0,1] -> HxWx3 uint8, approximating matplotlib's 'jet'."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def gradcam_overlay_base64(model, target_layer, input_tensor: torch.Tensor, display_image: Image.Image, class_idx: int) -> str:
    """
    Runs Grad-CAM for `class_idx` and returns a base64-encoded PNG data URL of
    the heatmap overlaid on `display_image` (resized to the model's input size).
    """
    cam_engine = GradCAM(model, target_layer)
    try:
        cam, _, _ = cam_engine(input_tensor, class_idx=class_idx)
    finally:
        cam_engine.remove_hooks()

    resized = display_image.convert("RGB").resize((input_tensor.shape[-1], input_tensor.shape[-2]))
    disp = np.array(resized)
    overlay = overlay_heatmap(disp, cam)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
