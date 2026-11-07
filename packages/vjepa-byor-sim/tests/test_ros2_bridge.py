"""The ROS 2 bridge node, end to end over real topics. Skipped without rclpy (run in the ROS 2 container)."""

import time

import pytest
import torch

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import Vector3Stamped  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import Image, JointState  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from vjepa_byor_sim import SimRig  # noqa: E402
from vjepa_byor_sim.ros2_bridge import LATCHED, SimBridge  # noqa: E402


@pytest.fixture
def ros():
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=99)  # apart from anything else running
    yield context
    rclpy.try_shutdown(context=context)


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class Harness:
    def __init__(self, context, start_mm=(40.0, 100.0)):
        self.bridge = SimBridge(
            context=context,
            parameter_overrides=[Parameter("start_mm", value=list(start_mm)), Parameter("rate_hz", value=0.0)],
        )
        self.client = rclpy.create_node("test_client", context=context)
        self.states: list[JointState] = []
        self.images: list[Image] = []
        self.client.create_subscription(JointState, "/rig/state", self.states.append, LATCHED)
        self.client.create_subscription(Image, "/camera/image_raw", self.images.append, LATCHED)
        self.cmd = self.client.create_publisher(Vector3Stamped, "/rig/cmd", 10)
        self.reset = self.client.create_client(Trigger, "/rig/reset")
        self.executor = SingleThreadedExecutor(context=context)
        self.executor.add_node(self.bridge)
        self.executor.add_node(self.client)

    def spin_until(self, condition, timeout=5.0):
        end = time.monotonic() + timeout
        while not condition():
            assert time.monotonic() < end, "timed out"
            self.executor.spin_once(timeout_sec=0.05)

    def send(self, dx_m, dy_m):
        """Sends a move and returns the answer: the first state stamped after the command."""
        self.spin_until(lambda: self.cmd.get_subscription_count() > 0)
        msg = Vector3Stamped()
        msg.header.stamp = self.client.get_clock().now().to_msg()
        msg.vector.x, msg.vector.y = dx_m, dy_m
        sent = ns(msg.header.stamp)
        self.cmd.publish(msg)
        self.spin_until(lambda: self.states and ns(self.states[-1].header.stamp) > sent
                        and self.images and ns(self.images[-1].header.stamp) > sent)
        return self.states[-1]

    def close(self):
        self.executor.shutdown()
        self.client.destroy_node()
        self.bridge.destroy_node()


@pytest.fixture
def harness(ros):
    h = Harness(ros)
    yield h
    h.close()


def photo(image: Image) -> torch.Tensor:
    return torch.frombuffer(bytearray(image.data), dtype=torch.uint8).view(image.height, image.width, 3).permute(2, 0, 1)


def test_late_subscriber_gets_the_current_state_and_photo(harness):
    harness.spin_until(lambda: harness.states and harness.images)
    state, image = harness.states[-1], harness.images[-1]
    assert list(state.name) == ["x", "y"]
    assert list(state.position) == pytest.approx([0.040, 0.100])
    assert (image.height, image.width, image.encoding, image.step) == (256, 256, "rgb8", 768)
    assert image.header.stamp == state.header.stamp


def test_a_command_moves_the_carriage_and_the_photo_matches_the_sim(harness):
    state = harness.send(0.020, -0.005)
    assert list(state.position) == pytest.approx([0.060, 0.095])
    image = harness.images[-1]
    assert image.header.stamp == state.header.stamp
    assert torch.equal(photo(image), SimRig(start_mm=(60, 95)).camera.capture())


def test_a_refused_move_is_still_answered_with_an_unchanged_state(harness):
    state = harness.send(-0.050, 0.0)  # would end at x = -10 mm
    assert list(state.position) == pytest.approx([0.040, 0.100])
    later = harness.send(-0.005, 0.0)
    assert list(later.position) == pytest.approx([0.035, 0.100])
    assert (later.header.stamp.sec, later.header.stamp.nanosec) > (state.header.stamp.sec, state.header.stamp.nanosec)


def test_reset_puts_the_carriage_back_at_the_start(harness):
    harness.send(0.020, 0.020)
    assert harness.reset.wait_for_service(timeout_sec=5.0)
    future = harness.reset.call_async(Trigger.Request())
    harness.spin_until(future.done)
    assert future.result().success
    harness.spin_until(lambda: list(harness.states[-1].position) == pytest.approx([0.040, 0.100]))


def test_periodic_republish_keeps_the_stamp_of_the_last_command(ros):
    h = Harness(ros)
    try:
        h.bridge.create_timer(0.01, h.bridge.republish)
        answer = h.send(0.005, 0.0)
        n = len(h.states)
        h.spin_until(lambda: len(h.states) >= n + 3)
        assert all(s.header.stamp == answer.header.stamp for s in h.states[n:])
        assert all(list(s.position) == pytest.approx([0.045, 0.100]) for s in h.states[n:])
    finally:
        h.close()
