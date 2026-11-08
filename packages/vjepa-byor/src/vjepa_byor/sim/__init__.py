"""The built-in ``sim`` rig: a simulated XY stage with a rendered top-down camera.

It has the same mechanics as the real stage (``vjepa-byor-esp32-stepper``), so collection,
fine-tuning and planning run end to end with no hardware. Built only from the library's public
building blocks, like any rig plugin. :mod:`vjepa_byor.sim.ros2_bridge` runs it as a ROS 2 robot.
"""

from vjepa_byor.sim.camera import DEFAULT_RESOLUTION, SimCamera
from vjepa_byor.sim.rig import DEFAULT_STEP_SIZES_MM, RIG_NAME, SimRig
from vjepa_byor.sim.stage import DEFAULT_TRAVEL_MM, SimStage, XYState

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
