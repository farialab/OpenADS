"""Lightweight 3D U-Net utilities for PWI lesion segmentation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# ----------------------------
# Utilities
# ----------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zscore_(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean()
    s = x.std()
    return (x - m) / (s + eps)


def resize3d(x_np: np.ndarray, size: Tuple[int, int, int], mode: str) -> torch.Tensor:
    x = torch.tensor(x_np, dtype=torch.float32)[None, None]  # [1,1,D,H,W]
    if mode == "trilinear":
        y = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    elif mode == "nearest":
        y = F.interpolate(x, size=size, mode="nearest")
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return y[0, 0]  # [D,H,W]


def dice_coeff_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum()
    denom = preds.sum() + targets.sum()
    return float((2 * inter + eps) / (denom + eps))

# ----------------------------
# Model
# ----------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, init_features=32):
        super().__init__()
        f = init_features
        self.enc1 = self._block(in_channels, f);  self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._block(f, f*2);         self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._block(f*2, f*4);       self.pool3 = nn.MaxPool3d(2)
        self.enc4 = self._block(f*4, f*8);       self.pool4 = nn.MaxPool3d(2)
        self.bottleneck = self._block(f*8, f*16)
        self.up4 = nn.ConvTranspose3d(f*16, f*8, 2, 2); self.dec4 = self._block(f*16, f*8)
        self.up3 = nn.ConvTranspose3d(f*8, f*4, 2, 2);  self.dec3 = self._block(f*8,  f*4)
        self.up2 = nn.ConvTranspose3d(f*4, f*2, 2, 2);  self.dec2 = self._block(f*4,  f*2)
        self.up1 = nn.ConvTranspose3d(f*2, f,   2, 2);  self.dec1 = self._block(f*2,  f)
        self.out = nn.Conv3d(f, out_channels, 1)

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    # def forward(self, x):
    #     e1 = self.enc1(x); p1 = self.pool1(e1)
    #     e2 = self.enc2(p1); p2 = self.pool2(e2)
    #     e3 = self.enc3(p2); p3 = self.pool3(e3)
    #     e4 = self.enc4(p3); p4 = self.pool4(e4)
    #     b  = self.bottleneck(p4)
    #     d4 = self.up4(b); d4 = torch.cat([d4, e4], 1); d4 = self.dec4(d4)
    #     d3 = self.up3(d4); d3 = torch.cat([d3, e3], 1); d3 = self.dec3(d3)
    #     d2 = self.up2(d3); d2 = torch.cat([d2, e2], 1); d2 = self.dec2(d2)
    #     d1 = self.up1(d2); d1 = torch.cat([d1, e1], 1); d1 = self.dec1(d1)
    #     return self.out(d1)
    def _align(self, x, ref):
        # x, ref: [N,C,D,H,W]; make x the same spatial size as ref
        if x.shape[-3:] != ref.shape[-3:]:
            x = F.interpolate(x, size=ref.shape[-3:], mode="trilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        b  = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = self._align(d4, e4)                # NEW
        d4 = torch.cat([d4, e4], 1); d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._align(d3, e3)                # NEW
        d3 = torch.cat([d3, e3], 1); d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._align(d2, e2)                # NEW
        d2 = torch.cat([d2, e2], 1); d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._align(d1, e1)                # NEW
        d1 = torch.cat([d1, e1], 1); d1 = self.dec1(d1)

        return self.out(d1)

def masked_bce_with_logits(logits, targets, mask, eps=1e-6):
    # Treat mask as per-voxel weight; normalize by masked voxels
    num = F.binary_cross_entropy_with_logits(
        logits, targets, weight=mask, reduction='sum'
    )
    den = mask.sum().clamp_min(eps)
    return num / den

def masked_soft_dice_loss(logits, targets, mask, eps=1e-6):
    p = torch.sigmoid(logits)
    p = p * mask
    t = targets * mask
    inter = (p * t).sum()
    denom = (p * p).sum() + (t * t).sum()
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice

def masked_combo_loss(logits, targets, mask, alpha=0.5):
    # Simple & reliable baseline: BCE + Dice
    bce = masked_bce_with_logits(logits, targets, mask)
    dsc = masked_soft_dice_loss(logits, targets, mask)
    return alpha * bce + (1.0 - alpha) * dsc

@torch.no_grad()
def masked_hard_dice_from_logits(logits, targets, mask, eps=1e-6):
    p = torch.sigmoid(logits)
    preds = (p > 0.5).float()
    preds = preds * mask
    t = targets * mask
    inter = (preds * t).sum()
    denom = preds.sum() + t.sum()
    return float((2.0 * inter + eps) / (denom + eps))
