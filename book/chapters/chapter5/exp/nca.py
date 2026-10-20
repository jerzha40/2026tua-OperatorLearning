"""Minimal Growing Neural Cellular Automata (NCA) research core.

Inspired by Mordvintsev et al. (2020), *Growing Neural Cellular Automata*.
This is an independent educational implementation, not their original code.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GrowingNCA(nn.Module):
    """Shared residual cell MLP + depthwise Sobel perception + asynchronous updates."""

    def __init__(self, channels: int = 12, hidden: int = 48, fire_rate: float = 0.5):
        super().__init__()
        if channels < 4 or not (0 < fire_rate <= 1):
            raise ValueError("channels>=4 and 0<fire_rate<=1 are required")
        self.channels, self.hidden, self.fire_rate = channels, hidden, fire_rate
        self.fc0 = nn.Conv2d(channels * 3, hidden, kernel_size=1)
        self.fc1 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fc1.weight)
        sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]) / 8
        sobel_y = sobel_x.T.contiguous()
        self.register_buffer('kernels', torch.stack([sobel_x,sobel_y]).unsqueeze(1))

    def seed(self, size: int, batch: int = 1, device: str | torch.device = "cpu") -> torch.Tensor:
        x = torch.zeros(batch, self.channels, size, size, device=device)
        x[:, 3:, size // 2, size // 2] = 1.0
        return x

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        # Convolve every state channel independently with Sobel X and Y.
        kernels = self.kernels.repeat(self.channels, 1, 1, 1)
        grad = F.conv2d(x, kernels, padding=1, groups=self.channels)
        # Depthwise outputs are (dx0,dy0, dx1,dy1, ...).
        b, _, h, w = grad.shape
        grad = grad.reshape(b, self.channels, 2, h, w)
        return torch.cat([x,grad[:,:,0],grad[:,:,1]], dim=1)

    @staticmethod
    def alive(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x: torch.Tensor, fire_rate: float | None = None) -> torch.Tensor:
        rate = self.fire_rate if fire_rate is None else fire_rate
        previous_alive = self.alive(x)
        dx = self.fc1(F.relu(self.fc0(self.perceive(x))))
        update_mask = (torch.rand_like(x[:, :1]) <= rate).to(x.dtype)
        x = x + dx * update_mask
        alive = previous_alive & self.alive(x)
        return x * alive.to(x.dtype)

    def rollout(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            x = self(x)
        return x

    @staticmethod
    def rgba(x: torch.Tensor) -> torch.Tensor:
        return x[:, :4].clamp(0, 1)


def flower_target(size: int = 32) -> torch.Tensor:
    """Procedural semi-transparent five-petal icon, CHW RGBA tensor in [0,1]."""
    yy, xx = np.mgrid[:size, :size]
    px = (xx + .5 - size/2) / (size/2)
    py = (yy + .5 - size/2) / (size/2)
    r = np.hypot(px, py)
    angle = np.arctan2(py, px)
    radius = .42 + .09 * np.cos(5*angle)
    alpha = 1 / (1 + np.exp(np.clip((r-radius)*55,-50,50)))
    center = np.exp(-((r/.16)**4))
    rgb = np.stack([.12 + .78*center, .78 - .34*center, .56 - .43*center], axis=0)
    return torch.from_numpy(np.concatenate([rgb*alpha[None],alpha[None]],axis=0).astype('float32'))


def render_rgba(x: torch.Tensor, background=(15,20,35)) -> np.ndarray:
    """Convert an NCA state to an RGB uint8 image on dark background."""
    arr = x.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    rgb = np.clip(arr[:3],0,1)
    a = np.clip(arr[3:4],0,1)
    bg = np.array(background,dtype='float32')[:,None,None] / 255
    result = rgb*a + bg*(1-a)
    return (np.clip(result,0,1).transpose(1,2,0)*255).astype('uint8')


def export_config(model: GrowingNCA) -> dict:
    return {"channels":model.channels,"hidden":model.hidden,"fire_rate":model.fire_rate}
