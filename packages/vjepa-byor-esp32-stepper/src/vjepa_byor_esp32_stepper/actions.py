"""The stage's action vocabulary (8 directions x 3 step sizes) and its XY state, in millimetres."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vjepa_byor.actions import DiscreteActionSpace, VectorState

DEFAULT_STEP_SIZES_MM = (1, 5, 20)


@dataclass(frozen=True)
class XYState(VectorState):
    """Carriage position in millimetres from the stage's home corner, as the ESP32 reports it."""

    x_mm: float
    y_mm: float

    def __post_init__(self) -> None:
        # JSON gives back ints for whole numbers; keep the fields floats either way
        object.__setattr__(self, "x_mm", float(self.x_mm))
        object.__setattr__(self, "y_mm", float(self.y_mm))


def xy_action_space(step_sizes_mm: Sequence[int] = DEFAULT_STEP_SIZES_MM) -> DiscreteActionSpace:
    """``[dx_mm, dy_mm]`` moves: N, NE, E, ... NW at each step size, labelled e.g. ``"NE 5mm"``.

    Step sizes are whole millimetres so every move survives the wire format exactly.
    """
    if any(not float(s).is_integer() for s in step_sizes_mm):
        raise ValueError(f"step sizes must be whole millimetres, got {list(step_sizes_mm)}")
    return DiscreteActionSpace.compass([int(s) for s in step_sizes_mm], unit="mm")
