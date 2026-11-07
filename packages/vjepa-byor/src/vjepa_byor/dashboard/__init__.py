"""The live demo dashboard: a browser page showing the camera, the goal photo, every move the
planner pictured, the error per step and the moves made, with recording and replay.

:class:`Hub` and :class:`DashboardServer` are plain Python; the ROS 2 side that feeds them is
:mod:`vjepa_byor.ros2.dashboard`.
"""

from vjepa_byor.dashboard.hub import Hub
from vjepa_byor.dashboard.web import Commands, DashboardServer

__all__ = ["Commands", "DashboardServer", "Hub"]
