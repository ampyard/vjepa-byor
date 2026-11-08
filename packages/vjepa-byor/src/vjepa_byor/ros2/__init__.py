"""The built-in ``ros2`` rig: any ROS 2 robot as a rig, described by a config file instead of a plugin.

    vjepa-byor collect --rig ros2 --rig-config apps/ros2/sim_robot.yaml --out data/ros2-sim ...

The file's format is described in :mod:`vjepa_byor.ros2.config`. ``rclpy`` comes from a sourced
ROS 2 environment (not from PyPI) and is imported only when the rig is opened, so building the rig
and checking its config work without ROS. The rest (numpy, PyYAML) comes with
``pip install vjepa-byor[ros2]``, in a venv that sees the ROS packages (``--system-site-packages``).
The rig is listed by ``available_rigs()`` only when ``rclpy`` is importable.

So far: commands on a topic (any message type with numeric fields), state from a topic
(``JointState`` joints or any numeric fields) and ``sensor_msgs/Image`` photos. ROS actions,
``CompressedImage``, a "state stopped changing" completion rule and rosbag import come later.
"""

from __future__ import annotations

from dataclasses import make_dataclass
from functools import cache
from typing import Any

from vjepa_byor.actions import VectorState
from vjepa_byor.rig import Rig
from vjepa_byor.ros2.config import (
    CameraConfig,
    CommandConfig,
    Ros2ConfigError,
    StateConfig,
    parse_actions,
    parse_mapping,
)
from vjepa_byor.ros2.link import Ros2ActionTransport, Ros2Camera, Ros2Link

RIG_NAME = "ros2"

__all__ = ["RIG_NAME", "Ros2ConfigError", "Ros2Rig", "state_type_for"]


def _floats(self: VectorState) -> None:
    for name in self.field_names():
        object.__setattr__(self, name, float(getattr(self, name)))


@cache
def state_type_for(names: tuple[str, ...]) -> type[VectorState]:
    """A frozen :class:`VectorState` with these float fields; the same class for the same names."""
    return make_dataclass(
        "Ros2State", [(n, float) for n in names], bases=(VectorState,), frozen=True,
        namespace={"__post_init__": _floats, "__module__": __name__},
    )


def Ros2Rig(
    camera: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    command: dict[str, Any] | None = None,
    actions: dict[str, Any] | None = None,
    mapping: dict[str, Any] | None = None,
    domain_id: int | None = None,
    connect_timeout: float = 10.0,
) -> Rig:
    """Builds the ``ros2`` rig from its config sections (the top-level keys of ``--rig-config``).

    Nothing connects to ROS until the rig is opened. Raises :class:`Ros2ConfigError` for a bad config.
    """
    sections = {"camera": camera, "state": state, "command": command, "actions": actions, "mapping": mapping}
    missing = [k for k, v in sections.items() if v is None]
    if missing:
        raise Ros2ConfigError(
            f"the ros2 rig needs a --rig-config with {', '.join(missing)} (see apps/ros2/sim_robot.yaml)"
        )
    camera_cfg = CameraConfig.parse(camera)
    state_cfg = StateConfig.parse(state)
    command_cfg = CommandConfig.parse(command)
    action_space = parse_actions(actions)
    if action_space.dim != len(command_cfg.fields):
        raise Ros2ConfigError(
            f"command.fields: {len(command_cfg.fields)} fields for {action_space.dim}-D actions; give one per dimension"
        )
    state_type = state_type_for(tuple(f.name for f in state_cfg.fields))
    if domain_id is not None and (isinstance(domain_id, bool) or not isinstance(domain_id, int) or domain_id < 0):
        raise Ros2ConfigError(f"domain_id: expected a ROS domain id like 42, got {domain_id!r}")

    link = Ros2Link(camera_cfg, state_cfg, command_cfg, state_type,
                    domain_id=domain_id, connect_timeout=float(connect_timeout))
    return Rig(
        name=RIG_NAME,
        action_space=action_space,
        state_type=state_type,
        mapping=parse_mapping(mapping, action_space.dim, len(state_cfg.fields)),
        transport=Ros2ActionTransport(link),
        camera=Ros2Camera(link),
        reply_timeout=command_cfg.timeout,
    )
