"""A JSON-per-line wire format, for rigs whose controller can parse JSON (or are simulated).

Commands and replies are single-line JSON objects::

    {"action": [5, -5]}             command: one action tensor
    {"state": {"x": 30, "y": 15}}   reply: the new state, as RigState.to_dict()
    {"error": "out of range"}       reply: the rig refused or failed
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch

from vjepa_byor.interfaces import CodecError, RigError, RigState


def _number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


class JsonLineCodec:
    """Implements :class:`~vjepa_byor.interfaces.CommandCodec` for any state type and action size."""

    def __init__(self, state_type: type[RigState], action_dim: int) -> None:
        self.state_type = state_type
        self.action_dim = action_dim

    @staticmethod
    def _parse(line: str) -> dict[str, Any]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise CodecError(f"not JSON: {line!r}") from e
        if not isinstance(data, dict):
            raise CodecError(f"not a JSON object: {line!r}")
        return data

    def encode_action(self, action: torch.Tensor) -> str:
        if action.shape != (self.action_dim,):
            raise ValueError(f"expected an action of shape ({self.action_dim},), got {tuple(action.shape)}")
        return json.dumps({"action": [_number(v) for v in action.tolist()]})

    def decode_action(self, line: str) -> torch.Tensor:
        values = self._parse(line).get("action")
        if not (
            isinstance(values, list)
            and len(values) == self.action_dim
            and all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in values)
        ):
            raise CodecError(f"bad command: {line!r}")
        return torch.tensor([float(v) for v in values])

    def encode_state(self, state: RigState) -> str:
        return json.dumps({"state": state.to_dict()})

    def decode_state(self, line: str) -> RigState:
        data = self._parse(line)
        if "error" in data:
            raise RigError(str(data["error"]))
        if not isinstance(data.get("state"), dict):
            raise CodecError(f"bad reply: {line!r}")
        try:
            return self.state_type.from_dict(data["state"])
        except (KeyError, TypeError, ValueError) as e:
            raise CodecError(f"bad state in reply {line!r}: {e}") from e
