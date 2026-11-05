"""Loads Meta's V-JEPA 2-AC (ViT-g encoder + action-conditioned predictor) and fine-tuned predictors.

The model code comes from the facebookresearch/vjepa2 repository through
``torch.hub``, pinned to one commit so upstream changes cannot break us; it
needs ``pip install vjepa-byor[vjepa]``. The weights are not bundled: they
have their own license. Download them once (about 11 GB) to
:func:`default_checkpoint`, or pass the path of your own copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vjepa_byor.predictor.input_layers import adapt_input_layers

HUB_REPO = "facebookresearch/vjepa2:204698b45b3712590f06245fbfba32d3be539812"
HUB_ENTRY = "vjepa2_ac_vit_giant"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
CHECKPOINT_NAME = "vjepa2-ac-vitg.pt"
MODEL_ID = f"{HUB_REPO}/{HUB_ENTRY}"

FINETUNED_WEIGHTS = "predictor.pt"
FINETUNED_INFO = "finetune.json"


def default_checkpoint() -> Path:
    """Where ``torch.hub`` keeps downloaded weights: ``~/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt``."""
    return Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT_NAME


@dataclass
class VJepaAC:
    """The two halves of V-JEPA 2-AC; either may be ``None`` if it was not asked for."""

    encoder: nn.Module | None
    predictor: nn.Module | None


def build_architecture() -> tuple[nn.Module, nn.Module]:
    """Encoder and predictor with random weights, built from the pinned hub code (about 10 s on CPU)."""
    try:
        import einops  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        raise ImportError("loading V-JEPA 2-AC needs its model code's dependencies: pip install vjepa-byor[vjepa]") from None
    return torch.hub.load(HUB_REPO, HUB_ENTRY, pretrained=False, trust_repo=True, verbose=False)


def _clean(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # same key clean-up as the upstream hubconf
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


def load_pretrained(
    checkpoint: str | Path | None = None, *, encoder: bool = True, predictor: bool = True
) -> VJepaAC:
    """The original V-JEPA 2-AC weights, on the CPU in float32 and in eval mode."""
    path = Path(checkpoint) if checkpoint is not None else default_checkpoint()
    if not path.is_file():
        raise FileNotFoundError(
            f"V-JEPA 2-AC weights not found at {path}. Download them (about 11 GB) from {CHECKPOINT_URL} "
            "to that path, or pass --checkpoint PATH"
        )
    enc, pred = build_architecture()
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if encoder:
        enc.load_state_dict(_clean(state["encoder"]))
    if predictor:
        pred.load_state_dict(_clean(state["predictor"]))
    del state
    return VJepaAC(enc.eval() if encoder else None, pred.eval() if predictor else None)


def save_finetuned(predictor: nn.Module, directory: str | Path, info: dict[str, Any]) -> Path:
    """Write a fine-tuned predictor's weights and a JSON description of how it was made."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / FINETUNED_WEIGHTS
    tmp = weights.with_suffix(".tmp")
    torch.save({k: v.detach().cpu() for k, v in predictor.state_dict().items()}, tmp)
    tmp.replace(weights)
    (directory / FINETUNED_INFO).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return weights


def read_finetuned_info(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / FINETUNED_INFO
    if not path.is_file():
        raise FileNotFoundError(f"{directory} is not a fine-tuned predictor (no {FINETUNED_INFO})")
    return json.loads(path.read_text(encoding="utf-8"))


def load_finetuned(directory: str | Path, predictor: nn.Module | None = None) -> nn.Module:
    """A predictor saved by :func:`save_finetuned`, in eval mode.

    ``predictor`` is the module to load into (any LoRA already merged); by
    default a fresh one from :func:`build_architecture`. Custom-sized input
    layers are recreated from the saved weights' shapes.
    """
    directory = Path(directory)
    read_finetuned_info(directory)
    state = torch.load(directory / FINETUNED_WEIGHTS, map_location="cpu", weights_only=True)
    if predictor is None:
        predictor = build_architecture()[1]
    adapt_input_layers(
        predictor, action_dim=state["action_encoder.weight"].shape[1], state_dim=state["state_encoder.weight"].shape[1]
    )
    predictor.load_state_dict(state)
    return predictor.eval()
