"""The simulated stage: an XY carriage that answers commands like a real rig's firmware.

It speaks the core library's JSON-line wire format
(:class:`vjepa_byor.actions.JsonLineCodec`) and is plugged in as the responder
of an in-memory :class:`vjepa_byor.testing.FakeTransport`, so the rig goes
through the same encode -> send -> reply -> decode path as hardware does.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from vjepa_byor import CodecError
from vjepa_byor.actions import JsonLineCodec, VectorState

# Same travel as the real stage, so sim and hardware datasets cover the same positions
DEFAULT_TRAVEL_MM = (200.0, 150.0)


@dataclass(frozen=True)
class XYState(VectorState):
    """Carriage position in millimetres from the stage's home corner (x right, y up)."""

    x_mm: float
    y_mm: float

    def __post_init__(self) -> None:
        # JSON gives back ints for whole numbers; keep the fields floats either way
        object.__setattr__(self, "x_mm", float(self.x_mm))
        object.__setattr__(self, "y_mm", float(self.y_mm))


class SimStage:
    """Moves the carriage by relative ``[dx_mm, dy_mm]`` actions and reports where it ends up.

    A move that would leave the travel area is refused and the carriage stays
    put, as on the real stage. Starts in the middle unless ``start_mm`` is given.
    """

    def __init__(self, travel_mm: Sequence[float] = DEFAULT_TRAVEL_MM, start_mm: Sequence[float] | None = None) -> None:
        self.width_mm, self.height_mm = (float(v) for v in travel_mm)
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError(f"travel must be positive, got {list(travel_mm)}")
        x, y = (self.width_mm / 2, self.height_mm / 2) if start_mm is None else (float(v) for v in start_mm)
        if not self.in_range(x, y):
            raise ValueError(f"start {[x, y]} is outside the travel area {[self.width_mm, self.height_mm]}")
        self.x_mm, self.y_mm = x, y
        self.codec = JsonLineCodec(XYState, action_dim=2)

    def in_range(self, x_mm: float, y_mm: float) -> bool:
        return 0 <= x_mm <= self.width_mm and 0 <= y_mm <= self.height_mm

    @property
    def state(self) -> XYState:
        return XYState(self.x_mm, self.y_mm)

    def __call__(self, line: str) -> str:
        try:
            dx, dy = self.codec.decode_action(line).tolist()
        except CodecError:
            return json.dumps({"error": "bad command"})
        x, y = self.x_mm + dx, self.y_mm + dy
        if not self.in_range(x, y):
            return json.dumps({"error": "out of range"})
        self.x_mm, self.y_mm = x, y
        return self.codec.encode_state(self.state)
