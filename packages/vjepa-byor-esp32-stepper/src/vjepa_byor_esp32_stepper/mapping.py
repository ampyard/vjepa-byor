"""Maps XY moves and positions to V-JEPA 2-AC's 7-D format: ``[dx, dy, 0, 0, 0, 0, 0]`` in metres."""

from __future__ import annotations

from vjepa_byor.actions import SlotMapping
from vjepa_byor.actions.vjepa_mapping import X, Y

MM_TO_M = 1e-3


def xy_mapping() -> SlotMapping:
    """The stage's x and y become the end effector's x and y; rotation, z and gripper stay zero.

    V-JEPA 2-AC was trained on end-effector motion in metres, so millimetres are
    converted to keep moves at the scale it has seen.
    """
    return SlotMapping([X, Y], [X, Y], action_scale=MM_TO_M, state_scale=MM_TO_M)
