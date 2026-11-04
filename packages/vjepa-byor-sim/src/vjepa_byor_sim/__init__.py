"""vjepa-byor rig plugin for a simulated XY stage, registered as ``sim``."""

from vjepa_byor_sim.camera import DEFAULT_RESOLUTION, SimCamera
from vjepa_byor_sim.rig import DEFAULT_STEP_SIZES_MM, RIG_NAME, SimRig
from vjepa_byor_sim.stage import DEFAULT_TRAVEL_MM, SimStage, XYState

__all__ = [
    "DEFAULT_RESOLUTION",
    "DEFAULT_STEP_SIZES_MM",
    "DEFAULT_TRAVEL_MM",
    "RIG_NAME",
    "SimCamera",
    "SimRig",
    "SimStage",
    "XYState",
]
