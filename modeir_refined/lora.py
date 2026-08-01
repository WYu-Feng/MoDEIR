from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.enabled = True
        self.scale = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(base.weight)
        self.lora_B.to(base.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int = 4, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        if base.groups != 1:
            raise ValueError("Grouped convolutions are not supported by encoder LoRA")
        self.base = base
        self.enabled = True
        self.scale = float(alpha) / float(r)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Conv2d(base.in_channels, r, 1, bias=False)
        self.lora_B = nn.Conv2d(
            r, base.out_channels, base.kernel_size, base.stride, base.padding, base.dilation, bias=False
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(base.weight)
        self.lora_B.to(base.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


class MultiLoRALinear(nn.Module):
    """Cross-attention LoRA bank with one independently selectable expert."""

    def __init__(self, base: nn.Linear, num_experts: int, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.num_experts = int(num_experts)
        self.r = int(r)
        self.lora_scale = float(alpha) / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.ModuleList([nn.Linear(base.in_features, self.r, bias=False) for _ in range(num_experts)])
        self.lora_B = nn.ModuleList([nn.Linear(self.r, base.out_features, bias=False) for _ in range(num_experts)])
        for a, b in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
            a.to(base.weight)
            b.to(base.weight)
        self.active_expert = 0
        self.active_scale: Optional[torch.Tensor] = None
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def set_active(self, expert_id: int, cond_scale: Optional[torch.Tensor] = None) -> None:
        self.active_expert = int(expert_id)
        self.active_scale = cond_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        delta = self.lora_B[self.active_expert](self.lora_A[self.active_expert](self.dropout(x))) * self.lora_scale
        scale = self.active_scale
        if scale is None:
            return base + delta
        scale = torch.as_tensor(scale, device=delta.device, dtype=delta.dtype)
        while scale.dim() < delta.dim():
            scale = scale.unsqueeze(1)
        return base + delta * scale


def inject_encoder_lora(encoder: nn.Module, r_lin: int = 8, a_lin: int = 8, r_conv: int = 8, a_conv: int = 8):
    replacements = []
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Linear):
            replacements.append((name, LoRALinear(module, r_lin, a_lin)))
        elif isinstance(module, nn.Conv2d) and module.groups == 1:
            replacements.append((name, LoRAConv2d(module, r_conv, a_conv)))
    for name, module in replacements:
        _set_module(encoder, name, module)


def inject_unet_expert_lora(unet: nn.Module, num_experts: int, r: int = 8, alpha: int = 16):
    replacements = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in ("to_q", "to_k", "to_v", "to_out")):
            replacements.append((name, MultiLoRALinear(module, num_experts, r, alpha)))
    for name, module in replacements:
        _set_module(unet, name, module)


def cache_expert_layers(unet: nn.Module) -> list[MultiLoRALinear]:
    return [module for module in unet.modules() if isinstance(module, MultiLoRALinear)]


def set_active_expert(layers: Iterable[MultiLoRALinear], expert_id: int) -> None:
    for layer in layers:
        layer.set_active(expert_id)
