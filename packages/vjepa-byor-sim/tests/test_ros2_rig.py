"""The built-in ros2 rig against the sim bridge node, over real topics (demo backlog item 5).

Skipped without rclpy: run in the ROS 2 container, ``docker/run.sh python -m pytest packages``.
"""

import time
from pathlib import Path

import pytest
import torch

rclpy = pytest.importorskip("rclpy")
yaml = pytest.importorskip("yaml")

from vjepa_byor import RigError, load_rig  # noqa: E402
from vjepa_byor.testing.collector_conformance import CollectorConformanceTests  # noqa: E402
from vjepa_byor.testing.conformance import RigConformanceTests  # noqa: E402
from vjepa_byor_sim import SimRig  # noqa: E402

from .conftest import BridgeThread  # noqa: E402

DOMAIN_ID = 98  # apart from the bridge tests and anything else running
SIM_ROBOT = Path(__file__).parents[3] / "apps" / "ros2" / "sim_robot.yaml"


@pytest.fixture
def bridge():
    b = BridgeThread(DOMAIN_ID)
    yield b
    b.close()


def ros2_rig(**overrides):
    config = yaml.safe_load(SIM_ROBOT.read_text())
    config.update(domain_id=DOMAIN_ID, **overrides)
    return load_rig("ros2", **config)


class TestRos2RigOnTheSimBridge(RigConformanceTests, CollectorConformanceTests):
    @pytest.fixture
    def rig(self, bridge):
        rig = ros2_rig()
        yield rig
        rig.close()

    @pytest.fixture
    def sample_states(self, rig):
        return [rig.state_type(0, 0), rig.state_type(100, 75), rig.state_type(12.5, 149.875)]


def test_a_move_returns_the_new_state_and_the_matching_photo(bridge):
    with ros2_rig() as rig:
        state = rig.execute(rig.action_space.by_label("NE 20mm"))
        assert state.to_dict() == pytest.approx({"x_mm": 120.0, "y_mm": 95.0})
        assert torch.equal(rig.camera.capture(), SimRig(start_mm=(120, 95)).camera.capture())


def test_a_refused_move_raises_and_the_next_one_works(bridge):
    with ros2_rig() as rig:
        for _ in range(5):
            rig.execute(rig.action_space.by_label("E 20mm"))  # 100 -> 200 mm, the edge
        with pytest.raises(RigError, match="refused"):
            rig.execute(rig.action_space.by_label("E 1mm"))
        assert rig.execute(rig.action_space.by_label("W 1mm")).to_dict() == pytest.approx({"x_mm": 199.0, "y_mm": 75.0})


def test_periodic_republish_is_never_taken_for_the_answer(bridge):
    # with 100 Hz republishing, a fresh stamp on the old position would often pass for the answer
    bridge.close()
    fast = BridgeThread(DOMAIN_ID, rate_hz=100.0)
    try:
        with ros2_rig() as rig:
            x = 100.0
            for _ in range(40):
                for label, dx in (("E 1mm", 1.0), ("W 1mm", -1.0)):
                    x += dx
                    assert rig.execute(rig.action_space.by_label(label)).x_mm == pytest.approx(x)
    finally:
        fast.close()


def test_no_robot_is_a_connection_error():
    rig = ros2_rig(connect_timeout=0.5)
    start = time.monotonic()
    with pytest.raises(ConnectionError, match="waiting for a subscriber on /rig/cmd and a state on /rig/state"):
        rig.open()
    assert time.monotonic() - start < 3
    assert not rig.transport.is_open


def test_unknown_message_field_is_reported_on_open(bridge):
    rig = ros2_rig(command={**yaml.safe_load(SIM_ROBOT.read_text())["command"],
                            "fields": [{"field": "vector.x"}, {"field": "vector.w"}]})
    with pytest.raises(ConnectionError, match="has no field 'vector.w'"):
        rig.open()
