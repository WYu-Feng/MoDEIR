from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bootstrap import PROJECT_ROOT


DEFAULT_DEGRADATION_LABELS = (
    "motion-blurry",
    "hazy",
    "jpeg-compressed",
    "low-light",
    "noisy",
    "raindrop",
    "rainy",
    "shadowed",
    "snowy",
    "uncompleted",
)


def _load_local_open_clip():
    module_name = "_modeir_daclip_open_clip"
    if module_name in sys.modules:
        return sys.modules[module_name]
    package_dir = PROJECT_ROOT / "da-clip" / "src" / "open_clip"
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(f"Cannot find local DA-CLIP open_clip package: {init_file}")
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import local DA-CLIP package from {init_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state:
        return state
    first_key = next(iter(state))
    if not first_key.startswith("module."):
        return state
    return {key[7:]: value for key, value in state.items()}


class DACLIPDegEncoder(nn.Module):
    """Frozen DA-CLIP degradation encoder used by the paper-style TAR router."""

    skip_checkpoint_state = True
    frozen_backbone = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "daclip_ViT-B-32",
        degradation_labels: Sequence[str] = DEFAULT_DEGRADATION_LABELS,
    ):
        super().__init__()
        self.checkpoint_path = str(checkpoint_path)
        self.model_name = str(model_name)
        self.degradation_labels = tuple(degradation_labels)
        open_clip = _load_local_open_clip()
        self.model = open_clip.create_model(self.model_name, pretrained=None, device="cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise TypeError(f"DA-CLIP checkpoint must contain a state dict: {self.checkpoint_path}")
        missing, unexpected = self.model.load_state_dict(_strip_module_prefix(state), strict=False)
        print(f"[LOAD] DA-CLIP: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        image_size = getattr(self.model.visual, "image_size", 224)
        if isinstance(image_size, (tuple, list)):
            self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            self.image_size = (int(image_size), int(image_size))
        mean = torch.tensor(getattr(self.model.visual, "image_mean", (0.48145466, 0.4578275, 0.40821073))).view(1, 3, 1, 1)
        std = torch.tensor(getattr(self.model.visual, "image_std", (0.26862954, 0.26130258, 0.27577711))).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean.float(), persistent=False)
        self.register_buffer("image_std", std.float(), persistent=False)

        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text = tokenizer(list(self.degradation_labels))
        with torch.no_grad():
            text_features = self.model.encode_text(text, normalize=True).float()
        self.register_buffer("degradation_text_features", text_features, persistent=False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if float(x.detach().amin()) < -0.05:
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(x, self.image_size, mode="bicubic", align_corners=False, antialias=True)
        return (x - self.image_mean.to(x)) / self.image_std.to(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_clip = self._preprocess(x)
        autocast_ctx = torch.autocast(device_type="cuda", enabled=False) if x_clip.device.type == "cuda" else nullcontext()
        with torch.no_grad(), autocast_ctx:
            image_feat, deg_feat = self.model.encode_image(x_clip.float(), control=True, normalize=True)
            deg_feat = deg_feat.float()
            image_feat = image_feat.float()
            text_features = self.degradation_text_features.to(device=deg_feat.device, dtype=deg_feat.dtype)
            scale = getattr(self.model, "logit_scale", None)
            logit_scale = scale.exp().float().clamp(max=100.0) if scale is not None else torch.tensor(100.0, device=deg_feat.device)
            deg_logits = logit_scale * (deg_feat @ text_features.t())
            deg_prob = F.softmax(deg_logits, dim=-1)
        return {
            "deg_feat": deg_feat,
            "deg_prob": deg_prob,
            "deg_logits": deg_logits.float(),
            "image_feat": image_feat,
        }
