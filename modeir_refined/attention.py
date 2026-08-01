from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


WindowMeta = Tuple[int, int, int, int, int, int]


def to_windows(x: torch.Tensor, window: int) -> tuple[torch.Tensor, WindowMeta]:
    """Convert BCHW features into non-overlapping windows."""
    b, c, h, w = x.shape
    ws = max(1, min(int(window), h, w))
    pad_h = (ws - h % ws) % ws
    pad_w = (ws - w % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    hp, wp = x.shape[-2:]
    x = x.view(b, c, hp // ws, ws, wp // ws, ws)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(b * (hp // ws) * (wp // ws), c, ws, ws), (b, h, w, hp, wp, ws)


def from_windows(x: torch.Tensor, meta: WindowMeta) -> torch.Tensor:
    """Restore BCHW features from non-overlapping windows."""
    b, h, w, hp, wp, ws = meta
    c = x.shape[1]
    x = x.view(b, hp // ws, wp // ws, c, ws, ws)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, hp, wp)
    return x[:, :, :h, :w]


def window_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    dim_head: int,
    window: int,
) -> torch.Tensor:
    """Apply scaled dot-product attention independently inside local windows."""
    qw, meta = to_windows(q, window)
    kw, _ = to_windows(k, window)
    vw, _ = to_windows(v, window)
    n, _, ws, _ = qw.shape

    def to_seq(x: torch.Tensor) -> torch.Tensor:
        x = x.view(n, heads, dim_head, ws * ws)
        return x.permute(0, 1, 3, 2).contiguous()

    out = F.scaled_dot_product_attention(to_seq(qw), to_seq(kw), to_seq(vw))
    out = out.permute(0, 1, 3, 2).contiguous()
    out = out.view(n, heads * dim_head, ws, ws)
    return from_windows(out, meta)
