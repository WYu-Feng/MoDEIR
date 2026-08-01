from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDegEncoder(nn.Module):
    def __init__(self, feat_dim: int = 512, prob_dim: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(256, 256, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_head = nn.Linear(256, feat_dim)
        self.prob_head = nn.Linear(256, prob_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x).flatten(1)
        return {"deg_feat": self.feat_head(h), "deg_prob": self.prob_head(h)}


class TARRouter(nn.Module):
    """Task-aware router with prototype severity matching and sparse top-s routing."""

    def __init__(
        self,
        num_experts: int,
        deg_feat_dim: int = 512,
        deg_prob_dim: int = 16,
        proj_dim: int = 256,
        top_s: int = 2,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_s = int(top_s)
        self.encoder = encoder or SimpleDegEncoder(deg_feat_dim, deg_prob_dim)
        self.proj_feat = nn.Sequential(nn.Linear(deg_feat_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.proj_type = nn.Sequential(nn.Linear(deg_prob_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, num_experts))
        self.prototypes = nn.Parameter(torch.randn(num_experts, proj_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(x)
        if not isinstance(out, dict) or "deg_feat" not in out or "deg_prob" not in out:
            raise TypeError("Degradation encoder must return deg_feat and deg_prob")
        return out

    def forward(self, x_deg: torch.Tensor, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        enc = self.extract(x_deg)
        feat_desc = F.normalize(self.proj_feat(enc["deg_feat"]), dim=1)
        sev_scores = feat_desc @ F.normalize(self.prototypes, dim=1).t()
        type_scores = self.proj_type(enc["deg_prob"])
        logits = sev_scores + type_scores + self.bias.view(1, -1)
        return {
            **enc,
            "feat_desc": feat_desc,
            "sev_scores": sev_scores,
            "type_scores": type_scores,
            "logits": logits,
            "probs": F.softmax(logits / temperature, dim=1),
        }

    @staticmethod
    def select(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=1)

    def route_topk(
        self,
        logits: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        *,
        training: bool = False,
        gumbel_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = max(1, min(int(self.top_s), int(logits.shape[1])))
        score_logits = logits
        if training and gumbel_noise:
            noise = -torch.empty_like(logits).exponential_().log()
            score_logits = logits + noise
        selected = torch.topk(score_logits, k=k, dim=1).indices
        dense = F.softmax(logits, dim=1) if probs is None else probs
        weights = dense.gather(1, selected)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return weights, selected

    @staticmethod
    def load_balance_loss(probs: torch.Tensor) -> torch.Tensor:
        usage = probs.mean(dim=0)
        return F.l1_loss(usage, torch.full_like(usage, 1.0 / usage.numel()))

    def severity_target(self, x_deg: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            deg = self.extract(x_deg)["deg_feat"]
            clean = self.extract(x_gt)["deg_feat"]
            target = F.normalize(deg - clean, dim=1)
        return target

    def severity_loss(self, route: dict[str, torch.Tensor], x_deg: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
        target = self.severity_target(x_deg, x_gt)
        if target.shape[1] != route["feat_desc"].shape[1]:
            with torch.no_grad():
                target = F.normalize(self.proj_feat(target), dim=1)
        return F.l1_loss(route["feat_desc"], target.to(route["feat_desc"]))
