"""Builds the simulated stage as a :class:`vjepa_byor.Rig`, from the core library's building blocks only."""

from __future__ import annotations

from collections.abc import Sequence

from vjepa_byor import Rig
from vjepa_byor.actions import DiscreteActionSpace, SlotMapping
from vjepa_byor.actions.vjepa_mapping import X, Y
from vjepa_byor.testing import FakeTransport
from vjepa_byor.sim.camera import DEFAULT_RESOLUTION, SimCamera
from vjepa_byor.sim.stage import DEFAULT_TRAVEL_MM, SimStage, XYState

RIG_NAME = "sim"

# Same vocabulary as the real stage: 8 compass directions x these step sizes
DEFAULT_STEP_SIZES_MM = (1, 5, 20)
MM_TO_M = 1e-3


def SimRig(
    step_sizes_mm: Sequence[float] = DEFAULT_STEP_SIZES_MM,
    travel_mm: Sequence[float] = DEFAULT_TRAVEL_MM,
    start_mm: Sequence[float] | None = None,
    resolution: Sequence[int] = DEFAULT_RESOLUTION,
    reply_timeout: float = 1.0,
) -> Rig:
    """A simulated XY stage: 8 directions x ``step_sizes_mm``, positions in millimetres.

    Registered as ``sim``; every argument can be set with ``--rig-config``.
    The stage is ``rig.transport.responder`` (also ``rig.camera.stage``).
    """
    stage = SimStage(travel_mm, start_mm)
    return Rig(
        name=RIG_NAME,
        action_space=DiscreteActionSpace.compass(list(step_sizes_mm), unit="mm"),
        state_type=XYState,
        # millimetres to metres in the end effector's x and y slots: [dx, dy, 0, 0, 0, 0, 0]
        mapping=SlotMapping([X, Y], [X, Y], action_scale=MM_TO_M, state_scale=MM_TO_M),
        codec=stage.codec,
        transport=FakeTransport(stage),
        camera=SimCamera(stage, resolution),
        reply_timeout=reply_timeout,
    )
