"""Rigs for the data collector tests.

``fake-sim`` is a small simulated rig kept separate from the built-in ``sim``, so the
collector tests do not depend on how ``sim`` renders: an XY gantry with 8 directions
x 3 step sizes, a JSON-line codec and a camera that draws the carriage. It is
registered as ``fake-sim``, never ``sim``, so it cannot clash with the built-in.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from vjepa_byor import CodecError, Rig, RigError
from vjepa_byor.testing import FakeTransport
from vjepa_byor.testing.fake_camera import MarkerCamera

# The toy line rig from ../conftest.py, with a camera that shows where the carriage is


@pytest.fixture
def marker_rig_factory(line_rig_factory):
    def make(start: int = 10) -> Rig:
        rig = line_rig_factory(start)
        device = rig.transport.responder
        rig.camera = MarkerCamera(lambda: [device.x], [(0, 20)])
        return rig

    return make


# fake-sim

WIDTH, HEIGHT = 60, 40  # travel in steps
DIRECTIONS = {"N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1), "S": (0, -1), "SW": (-1, -1), "W": (-1, 0), "NW": (-1, 1)}
STEP_SIZES = (1, 2, 5)


class XYState:
    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([float(self.x), float(self.y)])

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XYState:
        return cls(int(data["x"]), int(data["y"]))


class XYActions:
    dim = 2

    def __init__(self) -> None:
        self._rows = [(dx * s, dy * s) for s in STEP_SIZES for dx, dy in DIRECTIONS.values()]
        self._labels = [f"{name} {s}" for s in STEP_SIZES for name in DIRECTIONS]

    def actions(self) -> torch.Tensor:
        return torch.tensor(self._rows, dtype=torch.float32)

    def labels(self) -> list[str]:
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._rows)

    def contains(self, action: torch.Tensor) -> bool:
        return action.shape == (self.dim,) and bool((self.actions() == action).all(dim=1).any())


class XYMapping:
    action_dim = 7
    state_dim = 7

    def map_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return F.pad(actions, (0, 5))

    def map_states(self, states: torch.Tensor) -> torch.Tensor:
        return F.pad(states, (0, 5))


class JsonCodec:
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
        return json.dumps({"move": [int(v) for v in action.tolist()]})

    def decode_action(self, line: str) -> torch.Tensor:
        move = self._parse(line).get("move")
        if not (isinstance(move, list) and len(move) == 2):
            raise CodecError(f"bad command: {line!r}")
        return torch.tensor([float(v) for v in move])

    def encode_state(self, state: XYState) -> str:
        return json.dumps({"pos": [state.x, state.y]})

    def decode_state(self, line: str) -> XYState:
        data = self._parse(line)
        if "error" in data:
            raise RigError(data["error"])
        pos = data.get("pos")
        if not (isinstance(pos, list) and len(pos) == 2):
            raise CodecError(f"bad reply: {line!r}")
        return XYState(int(pos[0]), int(pos[1]))


class XYDevice:
    """Emulates the gantry's firmware: rejects moves that would leave the travel area."""

    def __init__(self, x: int = WIDTH // 2, y: int = HEIGHT // 2) -> None:
        self.x, self.y = x, y

    def __call__(self, line: str) -> str:
        dx, dy = (int(v) for v in JsonCodec().decode_action(line).tolist())
        if not (0 <= self.x + dx <= WIDTH and 0 <= self.y + dy <= HEIGHT):
            return json.dumps({"error": "out of range"})
        self.x, self.y = self.x + dx, self.y + dy
        return json.dumps({"pos": [self.x, self.y]})


def make_fake_sim(x: int = WIDTH // 2, y: int = HEIGHT // 2) -> Rig:
    device = XYDevice(x, y)
    return Rig(
        name="fake-sim",
        action_space=XYActions(),
        state_type=XYState,
        mapping=XYMapping(),
        codec=JsonCodec(),
        transport=FakeTransport(device),
        # one pixel per step or more, so every position gives a different photo
        camera=MarkerCamera(lambda: [device.x, device.y], [(0, WIDTH), (0, HEIGHT)], resolution=(48, 64)),
    )


@pytest.fixture(scope="session")
def fake_sim_factory():
    return make_fake_sim


@pytest.fixture
def fake_sim() -> Rig:
    return make_fake_sim()
