from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSobel(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        self.register_buffer("kx", kernel.view(1, 1, 3, 3))
        self.register_buffer("ky", kernel.t().view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x, self.kx.to(x), padding=1)
        gy = F.conv2d(x, self.ky.to(x), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)


def high_frequency_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sobel = FixedSobel().to(pred.device)
    return F.l1_loss(sobel(pred), sobel(target))


def critic_features(x: torch.Tensor, in_channels: int) -> torch.Tensor:
    if in_channels == 3:
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device, dtype=x.dtype)
        return F.conv2d(x, kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1), padding=1, groups=3)
    if in_channels == 9:
        kernels = torch.tensor(
            [[[1, 1], [-1, -1]], [[1, -1], [1, -1]], [[1, -1], [-1, 1]]],
            device=x.device,
            dtype=x.dtype,
        ) * 0.5
        return F.conv2d(x, kernels[:, None].repeat(3, 1, 1, 1), stride=2, groups=3)
    raise ValueError(f"Unsupported critic input channels: {in_channels}")


class NLayerPatchDiscriminator(nn.Module):
    """Architecture-compatible fixed Stage 2 critic."""

    def __init__(self, in_ch: int = 3, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        from torch.nn.utils import spectral_norm

        def conv(cin, cout, stride):
            return spectral_norm(nn.Conv2d(cin, cout, 4, stride, 1))

        layers: list[nn.Module] = [conv(in_ch, base_ch, 2), nn.LeakyReLU(0.2, inplace=True)]
        ch = base_ch
        for _ in range(1, n_layers):
            nxt = min(ch * 2, 512)
            layers.extend([conv(ch, nxt, 2), nn.InstanceNorm2d(nxt, affine=True), nn.LeakyReLU(0.2, inplace=True)])
            ch = nxt
        nxt = min(ch * 2, 512)
        layers.extend([conv(ch, nxt, 1), nn.InstanceNorm2d(nxt, affine=True), nn.LeakyReLU(0.2, inplace=True)])
        layers.append(conv(nxt, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gan_generator_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(-logits).mean()
