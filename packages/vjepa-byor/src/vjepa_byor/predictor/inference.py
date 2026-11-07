"""Predicts the next-state embedding for a whole basket of candidate actions in one batch.

This is what the planner calls: encode the current photo once, then run the
predictor for every action from the current rig state::

    model = ActionPredictor(encoder, predictor, rig.mapping)
    predicted = model.predict_basket(rig.camera.capture(), state, rig.action_space)  # (N, tokens, dim)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vjepa_byor.interfaces import ActionSpace, RigState, VJepaMapping
from vjepa_byor.predictor.encoding import autocast, encode_frames


def predict_next(predictor: nn.Module, z: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """One predictor step: ``z (B, tokens, dim)``, mapped ``actions`` and ``states`` ``(B, 1, d)`` -> the
    layer-normalised predicted latents of the next frame, ``(B, tokens, dim)``.

    The same step is used for fine-tuning, evaluation and planning.
    """
    out = predictor(z, actions, states)[:, -z.size(1) :]
    return F.layer_norm(out, (out.size(-1),))


@torch.inference_mode()
def predict_actions(
    predictor: nn.Module,
    mapping: VJepaMapping,
    z: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    """Predicted next latents ``float32 (N, tokens, dim)`` for each rig action in ``actions (N, action_dim)``,
    from the latents ``z (tokens, dim)`` of one photo and the rig state tensor ``state``.

    Runs on the predictor's device (bf16 on CUDA), ``batch_size`` actions at a time.
    """
    device = next(predictor.parameters()).device
    mapped_actions = mapping.map_actions(actions.float()).to(device)
    mapped_state = mapping.map_states(state.float()).to(device)
    z = z.to(device)
    out = []
    for start in range(0, len(mapped_actions), batch_size):
        a = mapped_actions[start : start + batch_size].unsqueeze(1)
        n = len(a)
        with autocast(device):
            pred = predict_next(predictor, z.expand(n, -1, -1), a, mapped_state.expand(n, 1, -1))
        out.append(pred.float())
    return torch.cat(out)


@torch.inference_mode()
def predict_pairs(
    predictor: nn.Module,
    mapping: VJepaMapping,
    z: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    """Predicted next latents ``float32 (N, tokens, dim)`` for N independent (latents, state, action) triples:
    ``z (N, tokens, dim)``, rig ``states (N, state_dim)`` and rig ``actions (N, action_dim)``.

    Used to roll the predictor forward from its own predictions (look-ahead planning).
    """
    device = next(predictor.parameters()).device
    mapped_actions = mapping.map_actions(actions.float()).to(device)
    mapped_states = mapping.map_states(states.float()).to(device)
    out = []
    for start in range(0, len(z), batch_size):
        end = start + batch_size
        with autocast(device):
            pred = predict_next(
                predictor,
                z[start:end].to(device),
                mapped_actions[start:end].unsqueeze(1),
                mapped_states[start:end].unsqueeze(1),
            )
        out.append(pred.float())
    return torch.cat(out)


class ActionPredictor:
    """The frozen encoder plus a (fine-tuned) predictor, ready to score actions.

    Both modules should already be on the device to run on; on CUDA the
    forward passes run in bf16. ``batch_size`` caps how many actions go through
    the predictor at once.
    """

    def __init__(self, encoder: nn.Module, predictor: nn.Module, mapping: VJepaMapping, *, batch_size: int = 64) -> None:
        self.encoder = encoder.eval()
        self.predictor = predictor.eval()
        self.mapping = mapping
        self.batch_size = batch_size
        self.device = next(predictor.parameters()).device

    def encode(self, photo: torch.Tensor) -> torch.Tensor:
        """A ``uint8 (3, H, W)`` photo -> its latents ``(tokens, dim)``."""
        return encode_frames(self.encoder, photo.unsqueeze(0))[0]

    def predict(self, z: torch.Tensor, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predicted next latents for each rig action in ``actions (N, action_dim)`` from rig state ``state``."""
        return predict_actions(self.predictor, self.mapping, z, state, actions, batch_size=self.batch_size)

    def predict_pairs(self, z: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predicted next latents for each row of ``z (N, tokens, dim)``, ``states`` and ``actions``."""
        return predict_pairs(self.predictor, self.mapping, z, states, actions, batch_size=self.batch_size)

    def predict_basket(self, photo: torch.Tensor, state: RigState, actions: torch.Tensor | ActionSpace) -> torch.Tensor:
        """Encode ``photo`` and predict the outcome of every action; rows follow ``actions``."""
        if not isinstance(actions, torch.Tensor):
            actions = actions.actions()
        return self.predict(self.encode(photo), state.as_tensor(), actions)
