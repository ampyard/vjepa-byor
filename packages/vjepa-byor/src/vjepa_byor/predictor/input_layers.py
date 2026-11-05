"""Optional custom-sized action/state input layers, for rigs that do not map naturally to 7-D.

V-JEPA 2-AC embeds each action and state with one linear layer from 7 inputs.
A :class:`~vjepa_byor.interfaces.VJepaMapping` with other sizes gets fresh
layers of the right width here; they start untrained, so a custom-sized rig
needs more fine-tuning data than one that reuses the pretrained 7-D layers.
"""

from __future__ import annotations

from torch import nn

ACTION_LAYER = "action_encoder"
STATE_LAYER = "state_encoder"


def adapt_input_layers(predictor: nn.Module, action_dim: int, state_dim: int, init_std: float = 0.02) -> list[str]:
    """Replace the predictor's action and state layers whose input size differs; returns the names replaced."""
    replaced = []
    for name, dim in ((ACTION_LAYER, action_dim), (STATE_LAYER, state_dim)):
        old: nn.Linear = getattr(predictor, name)
        if old.in_features == dim:
            continue
        new = nn.Linear(dim, old.out_features, bias=old.bias is not None)
        new.to(device=old.weight.device, dtype=old.weight.dtype)
        nn.init.trunc_normal_(new.weight, std=init_std)
        if new.bias is not None:
            nn.init.zeros_(new.bias)
        setattr(predictor, name, new)
        replaced.append(name)
    return replaced
