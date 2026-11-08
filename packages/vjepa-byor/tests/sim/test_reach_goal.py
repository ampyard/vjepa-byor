"""``/reach_goal`` end to end: action client -> server -> ros2 rig -> sim bridge (demo backlog item 6).

Runs in the ROS 2 container (needs rclpy and the vjepa_byor_interfaces package from the image).
The model is a stand-in that reads the carriage position from the photo, so these tests check the
server, not V-JEPA.
"""

import json
import threading
import time
from pathlib import Path

import pytest
import torch

rclpy = pytest.importorskip("rclpy")
yaml = pytest.importorskip("yaml")
pytest.importorskip("vjepa_byor_interfaces")

from action_msgs.msg import GoalStatus  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from vjepa_byor_interfaces.action import ReachGoal  # noqa: E402

from vjepa_byor import load_rig  # noqa: E402
from vjepa_byor.planning import PlannerConfig  # noqa: E402
from vjepa_byor.ros2.link import image_to_tensor, tensor_to_image  # noqa: E402
from vjepa_byor.ros2.reach_goal import LATCHED, ReachGoalServer  # noqa: E402
from vjepa_byor.sim import SimRig  # noqa: E402

from .conftest import BridgeThread  # noqa: E402

DOMAIN_ID = 97
SIM_ROBOT = Path(__file__).parents[4] / "apps" / "ros2" / "sim_robot.yaml"


def render(state):
    return SimRig(start_mm=state).camera.capture()


class CarriageModel:
    """Stands in for ActionPredictor: the latent is the carriage position in mm, read off the orange
    pixels, and predictions are exact. The L1 scorer's error is then half the Manhattan distance in mm."""

    def __init__(self, delay=0.0):
        self.delay = delay
        (ax, ay), (bx, by) = self._pixel((50, 40)), self._pixel((150, 110))
        self.scale = ((150 - 50) / (bx - ax), (110 - 40) / (by - ay))
        self.offset = (50 - ax * self.scale[0], 40 - ay * self.scale[1])

    @staticmethod
    def _pixel(state):
        photo = render(state).int()
        mask = (photo[0] > 200) & (photo[1] > 90) & (photo[1] < 170) & (photo[2] < 80)
        ys, xs = mask.nonzero(as_tuple=True)
        return xs.float().mean().item(), ys.float().mean().item()

    def encode(self, photo):
        photo = photo.int()
        mask = (photo[0] > 200) & (photo[1] > 90) & (photo[1] < 170) & (photo[2] < 80)
        ys, xs = mask.nonzero(as_tuple=True)
        x = xs.float().mean() * self.scale[0] + self.offset[0]
        y = ys.float().mean() * self.scale[1] + self.offset[1]
        return torch.tensor([[x.item(), y.item()]])

    def predict(self, z, state, actions):
        time.sleep(self.delay)
        return (state + actions).unsqueeze(1)


class Stack:
    """Bridge, ros2 rig, /reach_goal server and an action client, each node on its own thread."""

    def __init__(self, model):
        self.bridge = BridgeThread(DOMAIN_ID)
        config = yaml.safe_load(SIM_ROBOT.read_text())
        self.rig = load_rig("ros2", domain_id=DOMAIN_ID, **config)
        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=DOMAIN_ID)
        self.server = ReachGoalServer(model, self.rig, PlannerConfig(threshold=0.3, max_steps=30),
                                      render_goal=render, context=self.context)
        self.client_node = rclpy.create_node("test_client", context=self.context)
        self.client = ActionClient(self.client_node, ReachGoal, "/reach_goal")
        self.goal_images = []
        self.client_node.create_subscription(Image, "/reach_goal/goal_image", self.goal_images.append, LATCHED)
        self.executor = MultiThreadedExecutor(context=self.context)
        self.executor.add_node(self.server)
        self.executor.add_node(self.client_node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        assert self.client.wait_for_server(timeout_sec=10)

    def send(self, feedback=None, **goal):
        future = self.client.send_goal_async(ReachGoal.Goal(**goal), feedback_callback=feedback)
        wait(future.done)
        return future.result()

    def result(self, handle, timeout=30):
        future = handle.get_result_async()
        wait(future.done, timeout)
        return future.result()

    def close(self):
        self.executor.shutdown()
        self.thread.join(timeout=5)
        self.client.destroy()
        self.client_node.destroy_node()
        self.server.destroy_node()
        rclpy.try_shutdown(context=self.context)
        self.rig.close()
        self.bridge.close()


def wait(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


@pytest.fixture
def stack():
    s = Stack(CarriageModel())
    yield s
    s.close()


@pytest.fixture
def slow_stack():
    s = Stack(CarriageModel(delay=0.3))
    yield s
    s.close()


def test_reaches_a_goal_state_with_feedback_every_step(stack):
    feedback = []
    handle = stack.send(lambda msg: feedback.append(msg.feedback), goal_state=[127.0, 61.0])
    assert handle.accepted
    response = stack.result(handle)
    result = response.result
    assert response.status == GoalStatus.STATUS_SUCCEEDED
    assert (result.reached, result.reason) == (True, "reached")
    assert list(result.state_names) == ["x_mm", "y_mm"]
    assert list(result.final_state) == pytest.approx([127.0, 61.0])
    assert result.distance_to_goal == pytest.approx(0.0)
    assert result.moves >= 3  # 27 mm and 14 mm away with 1, 5 and 20 mm moves
    wait(lambda: len(feedback) == result.moves + 1)
    first, last = feedback[0], feedback[-1]
    assert first.step == 0 and list(first.state) == pytest.approx([100.0, 75.0])
    assert len(first.labels) == 24 and len(first.errors) == 24 and first.chosen in first.labels
    assert list(first.state_after) != list(first.state)
    assert last.chosen == "" and len(last.errors) == 0 and last.error <= 0.3
    wait(lambda: stack.goal_images)
    assert torch.equal(image_to_tensor(stack.goal_images[-1], (256, 256)), render([127, 61]))


def test_reaches_a_goal_photo(stack):
    photo = tensor_to_image(render([90, 80]), rclpy.clock.Clock().now().to_msg())
    response = stack.result(stack.send(goal_image=photo))
    assert response.status == GoalStatus.STATUS_SUCCEEDED
    assert list(response.result.final_state) == pytest.approx([90.0, 80.0])
    assert response.result.distance_to_goal == -1.0


@pytest.mark.parametrize(
    ("goal", "message"),
    [({}, "exactly one of goal_state and goal_image"),
     ({"goal_state": [100.0]}, "needs 2 values"),
     ({"goal_state": [250.0, 75.0]}, "outside the travel area")],
)
def test_bad_goals_are_aborted_with_the_reason(stack, goal, message):
    response = stack.result(stack.send(**goal))
    assert response.status == GoalStatus.STATUS_ABORTED
    assert message in response.result.reason and not response.result.reached


def test_max_steps_from_the_goal(stack):
    response = stack.result(stack.send(goal_state=[160.0, 30.0], max_steps=1))
    assert response.status == GoalStatus.STATUS_ABORTED
    assert (response.result.reason, response.result.moves) == ("max_steps", 1)


def test_cancel_stops_the_run_and_a_second_goal_waits_its_turn(slow_stack):
    s = slow_stack
    feedback = []
    handle = s.send(lambda msg: feedback.append(msg.feedback), goal_state=[180.0, 20.0])
    assert handle.accepted
    wait(lambda: feedback)
    assert not s.send(goal_state=[90.0, 80.0]).accepted  # busy
    cancel = handle.cancel_goal_async()
    wait(cancel.done)
    response = s.result(handle)
    assert response.status == GoalStatus.STATUS_CANCELED
    assert response.result.reason == "stopped"
    assert response.result.moves < 5
    assert s.send(goal_state=[100.0, 75.0]).accepted  # free again


# The dashboard's ROS side, on the same stack


def test_dashboard_sends_a_goal_and_records_the_whole_run(stack, tmp_path):
    from vjepa_byor.dashboard import Hub
    from vjepa_byor.ros2.config import CameraConfig, StateConfig
    from vjepa_byor.ros2.dashboard import DashboardNode

    config = yaml.safe_load(SIM_ROBOT.read_text())
    hub = Hub(tmp_path)
    node = DashboardNode(hub, CameraConfig.parse(config["camera"]), StateConfig.parse(config["state"]),
                         reset_service="/rig/reset", context=stack.context)
    stack.executor.add_node(node)
    try:
        wait(lambda: node.client.server_is_ready())
        wait(lambda: "camera" in hub._latest and "state" in hub._latest)
        assert hub._latest["camera"]["src"].startswith("data:image/png;base64,")
        assert node.commands.send_goal([110.0, 70.0]).startswith("goal sent")
        wait(lambda: hub.runs())
        events = [e for _, e in hub.load(hub.runs()[0])]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "camera" and "goal_image" in kinds
        assert kinds.count("start") == 1 and kinds[-1] == "end" and "step" in kinds
        start, end = events[kinds.index("start")], events[-1]
        assert start["goal_state"] == [110.0, 70.0] and len(start["labels"]) == 24
        assert end["status"] == "succeeded" and end["final_state"] == {"x_mm": 110.0, "y_mm": 70.0}
        wait(lambda: hub._latest["state"]["state"] == {"x_mm": 110.0, "y_mm": 70.0})
        # reset: back to the start
        assert node.commands.reset() == "resetting"
        wait(lambda: hub._latest["state"]["state"] == {"x_mm": 100.0, "y_mm": 75.0})
    finally:
        stack.executor.remove_node(node)
        node.destroy_node()


def test_dashboard_cancels_a_goal_sent_by_someone_else(slow_stack, tmp_path):
    from vjepa_byor.dashboard import Hub
    from vjepa_byor.ros2.config import CameraConfig, StateConfig
    from vjepa_byor.ros2.dashboard import DashboardNode

    s = slow_stack
    config = yaml.safe_load(SIM_ROBOT.read_text())
    node = DashboardNode(Hub(tmp_path), CameraConfig.parse(config["camera"]), StateConfig.parse(config["state"]),
                         context=s.context)
    s.executor.add_node(node)
    try:
        feedback = []
        handle = s.send(lambda msg: feedback.append(msg), goal_state=[180.0, 20.0])
        wait(lambda: feedback)
        wait(lambda: node.cancel_client.service_is_ready())
        assert node.commands.cancel() == "cancelling"
        assert s.result(handle).status == GoalStatus.STATUS_CANCELED
    finally:
        s.executor.remove_node(node)
        node.destroy_node()


# Step-by-step mode


def call_next(stack):
    from std_srvs.srv import Trigger

    client = getattr(stack, "next_client", None) or stack.client_node.create_client(Trigger, "/reach_goal/next")
    stack.next_client = client
    assert client.wait_for_service(timeout_sec=5)
    future = client.call_async(Trigger.Request())
    wait(future.done)
    return future.result()


def test_step_by_step_waits_for_next_before_every_move(stack):
    from std_msgs.msg import String

    events = []
    stack.client_node.create_subscription(String, "/reach_goal/events", lambda m: events.append(json.loads(m.data)), 50)
    feedback = []
    handle = stack.send(lambda msg: feedback.append(msg.feedback), goal_state=[127.0, 61.0], step_by_step=True)
    assert handle.accepted
    moves = 0
    while True:
        wait(lambda: any(e["type"] == "end" for e in events) or
             sum(e["type"] == "pending" for e in events) > moves)
        if any(e["type"] == "end" for e in events):
            break
        time.sleep(0.3)
        assert len(feedback) == moves  # nothing moved while it waits
        pending = [e for e in events if e["type"] == "pending"][-1]
        assert pending["step"] == moves and pending["chosen"] is None and len(pending["errors"]) == 24
        assert call_next(stack).success
        moves += 1
        wait(lambda: len(feedback) >= moves)  # after the last move, the final "at the goal" step may follow at once
        assert feedback[moves - 1].chosen == pending["planned"]  # it made the move it showed
    response = stack.result(handle)
    assert response.status == GoalStatus.STATUS_SUCCEEDED
    assert response.result.moves == moves >= 3
    assert not call_next(stack).success  # nothing waiting any more


def test_step_by_step_cancel_while_waiting(stack):
    feedback = []
    handle = stack.send(lambda msg: feedback.append(msg), goal_state=[127.0, 61.0], step_by_step=True)
    time.sleep(0.5)
    cancel = handle.cancel_goal_async()
    wait(cancel.done)
    response = stack.result(handle)
    assert response.status == GoalStatus.STATUS_CANCELED and response.result.moves == 0
    assert all(f.feedback.chosen == "" for f in feedback)  # only the final "stopped here" step, no move
