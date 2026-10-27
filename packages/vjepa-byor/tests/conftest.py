"""A toy rig on a 1-D rail, built only from the public interfaces.

It exercises the conformance suite, the Rig bundle and the registry without
any real plugin installed.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from vjepa_byor import CodecError, Rig, RigError
from vjepa_byor.testing import FakeTransport


class LineState:
    def __init__(self, x: int) -> None:
        self.x = x

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([float(self.x)])

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineState:
        return cls(int(data["x"]))


class LineActions:
    STEPS = (-5, -1, 1, 5)
    dim = 1

    def actions(self) -> torch.Tensor:
        return torch.tensor([[float(s)] for s in self.STEPS])

    def labels(self) -> list[str]:
        return [f"move {s:+d}" for s in self.STEPS]

    def __len__(self) -> int:
        return len(self.STEPS)

    def contains(self, action: torch.Tensor) -> bool:
        return action.shape == (self.dim,) and bool((self.actions() == action).all(dim=1).any())


class LineMapping:
    action_dim = 7
    state_dim = 7

    def map_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return F.pad(actions, (0, 6))

    def map_states(self, states: torch.Tensor) -> torch.Tensor:
        return F.pad(states, (0, 6))


class LineCodec:
    _MOVE = re.compile(r"MOVE (-?\d+)")
    _POS = re.compile(r"POS (-?\d+)")

    def encode_action(self, action: torch.Tensor) -> str:
        return f"MOVE {int(action[0].item())}"

    def decode_action(self, line: str) -> torch.Tensor:
        if not (m := self._MOVE.fullmatch(line)):
            raise CodecError(f"bad command: {line!r}")
        return torch.tensor([float(m[1])])

    def encode_state(self, state: LineState) -> str:
        return f"POS {state.x}"

    def decode_state(self, line: str) -> LineState:
        if line.startswith("ERR "):
            raise RigError(line[4:])
        if not (m := self._POS.fullmatch(line)):
            raise CodecError(f"bad reply: {line!r}")
        return LineState(int(m[1]))


class LineDevice:
    """Emulates the rig's firmware: a carriage on a rail from 0 to 20."""

    def __init__(self, x: int = 10) -> None:
        self.x = x

    def __call__(self, line: str) -> str:
        dx = int(LineCodec().decode_action(line)[0].item())
        if not 0 <= self.x + dx <= 20:
            return "ERR out of range"
        self.x += dx
        return f"POS {self.x}"


class FakeCamera:
    resolution = (32, 48)

    def capture(self) -> torch.Tensor:
        return torch.zeros((3, *self.resolution), dtype=torch.uint8)


def make_line_rig(start: int = 10) -> Rig:
    return Rig(
        name="line",
        action_space=LineActions(),
        state_type=LineState,
        mapping=LineMapping(),
        codec=LineCodec(),
        transport=FakeTransport(LineDevice(start)),
        camera=FakeCamera(),
    )


@pytest.fixture
def line_rig_factory():
    return make_line_rig


@pytest.fixture
def line_rig() -> Rig:
    return make_line_rig()


@pytest.fixture
def line_states() -> list[LineState]:
    return [LineState(0), LineState(7), LineState(20)]
