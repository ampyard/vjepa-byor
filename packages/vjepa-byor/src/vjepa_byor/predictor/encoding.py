"""Turns photos into the frozen encoder's latents, the same way for the latent cache and for inference.

Follows the upstream V-JEPA 2-AC evaluation: resize the shorter side to 256,
centre-crop 256 x 256, normalise with ImageNet statistics, repeat each frame to
fill the encoder's 2-frame tubelet, and layer-normalise the output tokens.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

CROP_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def autocast(device: torch.device, enabled: bool = True):
    """bf16 autocast on CUDA, nothing elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def preprocess(images: torch.Tensor, crop_size: int = CROP_SIZE) -> torch.Tensor:
    """``uint8 (N, 3, H, W)`` -> normalised ``float32 (N, 3, crop, crop)``."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected uint8 images of shape (N, 3, H, W), got {images.dtype} {tuple(images.shape)}")
    x = images.float()
    h, w = x.shape[-2:]
    if min(h, w) != crop_size:
        scale = crop_size / min(h, w)
        size = (max(crop_size, round(h * scale)), max(crop_size, round(w * scale)))
        x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        h, w = size
    top, left = (h - crop_size) // 2, (w - crop_size) // 2
    x = x[..., top : top + crop_size, left : left + crop_size]
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1) * 255
    return (x - mean) / std


@torch.no_grad()
def encode_frames(
    encoder: nn.Module, images: torch.Tensor, *, batch_size: int = 16, crop_size: int = CROP_SIZE
) -> torch.Tensor:
    """``uint8 (N, 3, H, W)`` photos -> layer-normalised ``float32 (N, tokens, dim)`` latents.

    Runs on the encoder's device, in bf16 on CUDA. Each photo is encoded on its
    own, so results do not depend on ``batch_size``.
    """
    device = next(encoder.parameters()).device
    out = []
    for start in range(0, len(images), batch_size):
        x = preprocess(images[start : start + batch_size].to(device), crop_size)
        clip = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (B, 3, 2, crop, crop): one frame fills the tubelet
        with autocast(device):
            h = encoder(clip)
        out.append(F.layer_norm(h.float(), (h.size(-1),)))
    return torch.cat(out)
