"""The ``ros2`` rig's config file, parsed and checked without ROS (so it is testable anywhere).

Every top-level key is an option of :func:`vjepa_byor.ros2.Ros2Rig`, so the file goes straight to
``--rig-config``. See ``apps/ros2/sim_robot.yaml`` for a complete example::

    camera:
      topic: /camera/image_raw          # sensor_msgs/msg/Image (rgb8, bgr8, rgba8, bgra8 or mono8)
      resolution: [256, 256]            # [height, width]; other sizes are resized
    state:
      topic: /rig/state
      type: sensor_msgs/msg/JointState
      fields:                           # the rig state, in order
        - {name: x_mm, joint: x, scale: 1000}           # JointState: a joint's position
        - {name: y_mm, field: pose.position.y}          # any message: a dotted field path
    command:
      topic: /rig/cmd
      type: geometry_msgs/msg/Vector3Stamped
      fields:                           # one per action dimension: field = action[i] * scale
        - {field: vector.x, scale: 0.001}
        - {field: vector.y, scale: 0.001}
    actions: {compass: [1, 5, 20], unit: mm}            # or {rows: [[...], ...], labels: [...]}
    mapping: {action_slots: [x, y], state_slots: [x, y], action_scale: 0.001, state_scale: 0.001}

A move counts as finished at the first state stamped (``header.stamp``) after the command was sent,
so the robot must publish its state after handling each command; a state that does not change
means the move was refused (``refused_if_unchanged``). The photo used is the first one stamped no
earlier than that state. ``qos`` is ``reliable`` (the default), ``latched`` (transient local, for
publishers that latch) or ``best_effort`` (sensor drivers).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vjepa_byor.actions import DiscreteActionSpace, SlotMapping
from vjepa_byor.actions.vjepa_mapping import VJEPA_AC_DIM

QOS_PROFILES = ("reliable", "latched", "best_effort")
SLOT_NAMES = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5, "gripper": 6}
_MSG_TYPE = re.compile(r"^[A-Za-z_][\w]*/msg/[A-Z]\w*$")
_FIELD_PATH = re.compile(r"^[a-z_]\w*(\.[a-z_]\w*)*$")
_STATE_NAME = re.compile(r"^[a-z_]\w*$")


class Ros2ConfigError(ValueError):
    """The ``ros2`` rig's config is missing something or has a wrong value."""


def _require(data: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise Ros2ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    return data


def _keys(data: Mapping[str, Any], where: str, required: Sequence[str], optional: Sequence[str] = ()) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise Ros2ConfigError(f"{where}: missing {', '.join(missing)}")
    unknown = sorted(set(data) - set(required) - set(optional))
    if unknown:
        raise Ros2ConfigError(f"{where}: unknown key(s) {', '.join(unknown)}")


def _topic(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise Ros2ConfigError(f"{where}.topic: expected an absolute topic name like /rig/state, got {value!r}")
    return value


def _msg_type(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _MSG_TYPE.match(value):
        raise Ros2ConfigError(f"{where}.type: expected a message type like sensor_msgs/msg/JointState, got {value!r}")
    return value


def _qos(value: Any, where: str) -> str:
    if value not in QOS_PROFILES:
        raise Ros2ConfigError(f"{where}.qos: expected one of {', '.join(QOS_PROFILES)}, got {value!r}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Ros2ConfigError(f"{where}: expected a number, got {value!r}")
    if positive and value <= 0:
        raise Ros2ConfigError(f"{where}: must be positive, got {value!r}")
    return float(value)


def _field_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _FIELD_PATH.match(value):
        raise Ros2ConfigError(f"{where}: expected a dotted field path like vector.x, got {value!r}")
    return value


@dataclass(frozen=True)
class CameraConfig:
    topic: str
    resolution: tuple[int, int] = (256, 256)
    qos: str = "reliable"
    timeout: float = 5.0

    @classmethod
    def parse(cls, data: Any) -> CameraConfig:
        data = _require(data, "camera")
        _keys(data, "camera", ["topic"], ["resolution", "qos", "timeout"])
        resolution = data.get("resolution", [256, 256])
        if (
            not isinstance(resolution, Sequence)
            or len(resolution) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in resolution)
        ):
            raise Ros2ConfigError(f"camera.resolution: expected [height, width] in pixels, got {resolution!r}")
        return cls(
            topic=_topic(data["topic"], "camera"),
            resolution=(resolution[0], resolution[1]),
            qos=_qos(data.get("qos", "reliable"), "camera"),
            timeout=_number(data.get("timeout", 5.0), "camera.timeout", positive=True),
        )


@dataclass(frozen=True)
class StateField:
    """One number of the rig state: a joint's position (``JointState``) or a message field, times ``scale``."""

    name: str
    joint: str | None = None
    field: str | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class StateConfig:
    topic: str
    type: str
    fields: tuple[StateField, ...]
    qos: str = "reliable"

    @classmethod
    def parse(cls, data: Any) -> StateConfig:
        data = _require(data, "state")
        _keys(data, "state", ["topic", "type", "fields"], ["qos"])
        msg_type = _msg_type(data["type"], "state")
        raw = data["fields"]
        if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
            raise Ros2ConfigError("state.fields: expected a non-empty list")
        fields = []
        for i, item in enumerate(raw):
            where = f"state.fields[{i}]"
            item = _require(item, where)
            _keys(item, where, ["name"], ["joint", "field", "scale"])
            name = item["name"]
            if not isinstance(name, str) or not _STATE_NAME.match(name):
                raise Ros2ConfigError(f"{where}.name: expected a lowercase identifier like x_mm, got {name!r}")
            if ("joint" in item) == ("field" in item):
                raise Ros2ConfigError(f"{where}: give exactly one of joint (for a JointState) or field")
            if "joint" in item:
                if msg_type != "sensor_msgs/msg/JointState":
                    raise Ros2ConfigError(f"{where}.joint: only works with sensor_msgs/msg/JointState; use field")
                if not isinstance(item["joint"], str) or not item["joint"]:
                    raise Ros2ConfigError(f"{where}.joint: expected a joint name, got {item['joint']!r}")
            fields.append(
                StateField(
                    name=name,
                    joint=item.get("joint"),
                    field=_field_path(item["field"], f"{where}.field") if "field" in item else None,
                    scale=_number(item.get("scale", 1.0), f"{where}.scale"),
                )
            )
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise Ros2ConfigError("state.fields: names must be unique")
        return cls(
            topic=_topic(data["topic"], "state"),
            type=msg_type,
            fields=tuple(fields),
            qos=_qos(data.get("qos", "reliable"), "state"),
        )


@dataclass(frozen=True)
class CommandField:
    """Where one action dimension goes in the command message: ``field = action[i] * scale``."""

    field: str
    scale: float = 1.0


@dataclass(frozen=True)
class CommandConfig:
    topic: str
    type: str
    fields: tuple[CommandField, ...]
    timeout: float = 5.0
    refused_if_unchanged: bool = True
    tolerance: float = 1e-9

    @classmethod
    def parse(cls, data: Any) -> CommandConfig:
        data = _require(data, "command")
        _keys(data, "command", ["topic", "type", "fields"], ["timeout", "refused_if_unchanged", "tolerance"])
        raw = data["fields"]
        if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
            raise Ros2ConfigError("command.fields: expected a non-empty list, one entry per action dimension")
        fields = []
        for i, item in enumerate(raw):
            where = f"command.fields[{i}]"
            item = _require(item, where)
            _keys(item, where, ["field"], ["scale"])
            scale = _number(item.get("scale", 1.0), f"{where}.scale")
            if scale == 0:
                raise Ros2ConfigError(f"{where}.scale: must not be zero")
            fields.append(CommandField(field=_field_path(item["field"], f"{where}.field"), scale=scale))
        paths = [f.field for f in fields]
        if len(set(paths)) != len(paths):
            raise Ros2ConfigError("command.fields: each message field can be filled only once")
        if "header" in {p.split(".")[0] for p in paths}:
            raise Ros2ConfigError("command.fields: the header is filled in automatically")
        refused = data.get("refused_if_unchanged", True)
        if not isinstance(refused, bool):
            raise Ros2ConfigError(f"command.refused_if_unchanged: expected true or false, got {refused!r}")
        tolerance = _number(data.get("tolerance", 1e-9), "command.tolerance")
        if tolerance < 0:
            raise Ros2ConfigError("command.tolerance: must not be negative")
        return cls(
            topic=_topic(data["topic"], "command"),
            type=_msg_type(data["type"], "command"),
            fields=tuple(fields),
            timeout=_number(data.get("timeout", 5.0), "command.timeout", positive=True),
            refused_if_unchanged=refused,
            tolerance=tolerance,
        )


def parse_actions(data: Any) -> DiscreteActionSpace:
    """``{compass: [1, 5, 20], unit: mm}`` (8 directions x step sizes) or ``{rows: [...], labels: [...]}``."""
    data = _require(data, "actions")
    try:
        if "compass" in data:
            _keys(data, "actions", ["compass"], ["unit"])
            steps = data["compass"]
            if not isinstance(steps, Sequence) or isinstance(steps, str):
                raise Ros2ConfigError("actions.compass: expected a list of step sizes")
            return DiscreteActionSpace.compass(list(steps), unit=str(data.get("unit", "")))
        _keys(data, "actions", ["rows", "labels"])
        return DiscreteActionSpace(data["rows"], data["labels"])
    except Ros2ConfigError:
        raise
    except (TypeError, ValueError) as e:
        raise Ros2ConfigError(f"actions: {e}") from e


def _slots(value: Any, where: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise Ros2ConfigError(f"{where}: expected a list of slots such as [x, y]")
    slots = []
    for v in value:
        if isinstance(v, str) and v in SLOT_NAMES:
            slots.append(SLOT_NAMES[v])
        elif isinstance(v, int) and not isinstance(v, bool) and 0 <= v < VJEPA_AC_DIM:
            slots.append(v)
        else:
            raise Ros2ConfigError(f"{where}: {v!r} is not one of {', '.join(SLOT_NAMES)} (or 0-{VJEPA_AC_DIM - 1})")
    return slots


def parse_mapping(data: Any, action_dim: int, state_dim: int) -> SlotMapping:
    data = _require(data, "mapping")
    _keys(data, "mapping", ["action_slots", "state_slots"], ["action_scale", "state_scale", "state_offset"])
    action_slots = _slots(data["action_slots"], "mapping.action_slots")
    state_slots = _slots(data["state_slots"], "mapping.state_slots")
    if len(action_slots) != action_dim:
        raise Ros2ConfigError(f"mapping.action_slots: {len(action_slots)} slots for {action_dim}-D actions")
    if len(state_slots) != state_dim:
        raise Ros2ConfigError(f"mapping.state_slots: {len(state_slots)} slots for {state_dim} state fields")
    try:
        return SlotMapping(
            action_slots,
            state_slots,
            action_scale=data.get("action_scale", 1.0),
            state_scale=data.get("state_scale", 1.0),
            state_offset=data.get("state_offset", 0.0),
        )
    except (TypeError, ValueError) as e:
        raise Ros2ConfigError(f"mapping: {e}") from e
