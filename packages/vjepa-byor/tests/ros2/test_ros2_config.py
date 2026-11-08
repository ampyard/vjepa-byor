"""The ros2 rig without ROS: config checking, the state type and photo conversion."""

import copy
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vjepa_byor import ActionTransport, Camera, available_rigs, load_rig, registry
from vjepa_byor.ros2 import Ros2ConfigError, Ros2Rig, state_type_for
from vjepa_byor.ros2.link import image_to_tensor

yaml = pytest.importorskip("yaml")

SIM_ROBOT = Path(__file__).parents[4] / "apps" / "ros2" / "sim_robot.yaml"


@pytest.fixture
def config():
    return yaml.safe_load(SIM_ROBOT.read_text())


def test_ros2_is_listed_only_when_rclpy_is_importable():
    assert ("ros2" in available_rigs()) == (find_spec("rclpy") is not None)


def test_ros2_is_listed_when_rclpy_is_importable(monkeypatch):
    monkeypatch.setattr(registry, "find_spec", lambda module: object())
    assert "ros2" in available_rigs()


def test_ros2_is_hidden_but_still_loadable_without_rclpy(monkeypatch, config):
    monkeypatch.setattr(registry, "find_spec", lambda module: None)
    assert "ros2" not in available_rigs()
    assert load_rig("ros2", **config).name == "ros2"


def test_sim_robot_yaml_builds_the_same_actions_and_mapping_as_the_sim(config):
    rig = load_rig("ros2", **config)
    assert rig.name == "ros2" and not rig.uses_lines
    assert isinstance(rig.transport, ActionTransport) and isinstance(rig.camera, Camera)
    assert len(rig.action_space) == 24
    assert rig.action_space.by_label("SW 20mm").tolist() == [-20.0, -20.0]
    mapped = rig.mapping.map_actions(rig.action_space.by_label("NE 5mm"))
    assert mapped.tolist() == pytest.approx([0.005, 0.005, 0, 0, 0, 0, 0])
    assert rig.camera.resolution == (256, 256)
    assert rig.reply_timeout == 5.0
    assert not rig.transport.is_open  # nothing connects before open()


def test_state_type_has_the_configured_fields(config):
    rig = Ros2Rig(**config)
    state = rig.state_type.from_dict({"x_mm": 100, "y_mm": 75})
    assert state.to_dict() == {"x_mm": 100.0, "y_mm": 75.0}
    assert state.as_tensor().tolist() == [100.0, 75.0]
    assert state_type_for(("x_mm", "y_mm")) is rig.state_type


def test_no_config_explains_what_is_needed():
    with pytest.raises(Ros2ConfigError, match="needs a --rig-config with camera, state, command"):
        load_rig("ros2")


def broken(config, path, value):
    config = copy.deepcopy(config)
    *parents, last = path
    target = config
    for key in parents:
        target = target[key]
    if value is KeyError:
        del target[last]
    else:
        target[last] = value
    return config


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("camera", "topic"), "camera/image_raw", "absolute topic"),
        (("camera", "resolution"), [256], r"\[height, width\]"),
        (("camera", "qos"), "fast", "one of reliable, latched, best_effort"),
        (("camera", "fps"), 30, "unknown key"),
        (("state", "type"), "JointState", "message type like"),
        (("state", "fields"), [], "non-empty list"),
        (("state", "fields", 0, "joint"), KeyError, "exactly one of joint"),
        (("state", "fields", 1, "name"), "x_mm", "unique"),
        (("state", "fields", 0, "name"), "X mm", "lowercase identifier"),
        (("command", "fields", 0, "scale"), 0, "must not be zero"),
        (("command", "fields", 1, "field"), "vector.x", "only once"),
        (("command", "fields", 0, "field"), "header.frame_id", "header is filled in"),
        (("command", "timeout"), -1, "positive"),
        (("command", "refused_if_unchanged"), "yes", "true or false"),
        (("command", "topic"), KeyError, "missing topic"),
        (("actions", "compass"), [0, 5], "step sizes must be positive"),
        (("mapping", "action_slots"), ["x", "q"], "'q' is not one of"),
        (("mapping", "state_slots"), ["x"], "1 slots for 2 state fields"),
    ],
)
def test_bad_config_is_reported_with_its_place(config, path, value, message):
    with pytest.raises(Ros2ConfigError, match=message):
        Ros2Rig(**broken(config, path, value))


def test_joint_needs_a_joint_state(config):
    config["state"]["type"] = "geometry_msgs/msg/PoseStamped"
    with pytest.raises(Ros2ConfigError, match="only works with sensor_msgs/msg/JointState"):
        Ros2Rig(**config)


def test_one_command_field_per_action_dimension(config):
    config["command"]["fields"].append({"field": "vector.z"})
    with pytest.raises(Ros2ConfigError, match="3 fields for 2-D actions"):
        Ros2Rig(**config)


def test_explicit_action_rows(config):
    config["actions"] = {"rows": [[1, 0], [-1, 0]], "labels": ["right", "left"]}
    assert Ros2Rig(**config).action_space.labels() == ["right", "left"]


def image(encoding, pixels, step_pad=0):
    h, w, c = pixels.shape
    rows = torch.cat([pixels.reshape(h, w * c), torch.zeros(h, step_pad, dtype=torch.uint8)], dim=1)
    return SimpleNamespace(encoding=encoding, height=h, width=w, step=w * c + step_pad, data=rows.numpy().tobytes())


def test_photo_conversion_handles_bgr_and_row_padding():
    rgb = torch.randint(0, 256, (4, 6, 3), dtype=torch.uint8)
    expected = rgb.permute(2, 0, 1)
    assert torch.equal(image_to_tensor(image("rgb8", rgb), (4, 6)), expected)
    assert torch.equal(image_to_tensor(image("bgr8", rgb.flip(-1), step_pad=2), (4, 6)), expected)
    gray = torch.randint(0, 256, (4, 6, 1), dtype=torch.uint8)
    assert torch.equal(image_to_tensor(image("mono8", gray), (4, 6)), gray.permute(2, 0, 1).expand(3, -1, -1))


def test_photo_is_resized_to_the_configured_resolution():
    flat = torch.full((8, 12, 3), 77, dtype=torch.uint8)
    out = image_to_tensor(image("rgb8", flat), (4, 6))
    assert out.shape == (3, 4, 6) and out.dtype == torch.uint8 and (out == 77).all()


def test_unsupported_encoding():
    with pytest.raises(ValueError, match="unsupported image encoding"):
        image_to_tensor(image("16UC1", torch.zeros(2, 2, 2, dtype=torch.uint8)), (2, 2))
