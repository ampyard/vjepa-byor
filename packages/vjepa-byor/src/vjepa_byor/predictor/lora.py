"""Low-rank adapters on the predictor, the fallback if full fine-tuning does not fit in VRAM.

:func:`apply_lora` wraps the attention and MLP layers of every predictor block
in a :class:`LoRALinear` and freezes everything else; only the small ``A`` and
``B`` matrices train. :func:`merge_lora` folds them back into plain
``nn.Linear`` layers, so a fine-tuned predictor is saved and loaded the same
way whichever method made it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

DEFAULT_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class LoRALinear(nn.Module):
    """``base(x) + (x A^T B^T) * alpha / rank``, with ``base`` frozen and ``B`` starting at zero."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        linear = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        linear.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        linear.weight.copy_(self.base.weight + (self.lora_b @ self.lora_a) * self.scale)
        if self.base.bias is not None:
            linear.bias.copy_(self.base.bias)
        return linear


def _replace(root: nn.Module, dotted: str, module: nn.Module) -> None:
    parent_name, _, child = dotted.rpartition(".")
    setattr(root.get_submodule(parent_name) if parent_name else root, child, module)


def apply_lora(
    predictor: nn.Module,
    rank: int,
    alpha: float | None = None,
    targets: Sequence[str] = DEFAULT_TARGETS,
    blocks: str = "predictor_blocks",
) -> int:
    """Freeze ``predictor`` and add adapters to ``targets`` in each of its ``blocks``; returns how many."""
    predictor.requires_grad_(False)
    count = 0
    for block in predictor.get_submodule(blocks):
        for target in targets:
            layer = block.get_submodule(target)
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"{target} is a {type(layer).__name__}, not nn.Linear")
            _replace(block, target, LoRALinear(layer, rank, alpha))
            count += 1
    return count


def merge_lora(predictor: nn.Module) -> int:
    """Replace every :class:`LoRALinear` with its merged ``nn.Linear``; returns how many."""
    wrapped = [name for name, m in predictor.named_modules() if isinstance(m, LoRALinear)]
    for name in wrapped:
        _replace(predictor, name, predictor.get_submodule(name).merged())
    return len(wrapped)
