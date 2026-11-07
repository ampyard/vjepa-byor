"""Feeds the dashboard from ROS 2: camera, goal photo, robot state and ``/reach_goal`` run events.

It also sends goals to ``/reach_goal`` (a click on the mat), cancels them (any goal, also one sent
from a terminal) and calls the robot's reset service. Needs ``rclpy`` and the interfaces package.
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

from action_msgs.srv import CancelGoal
from PIL import Image as PILImage
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vjepa_byor_interfaces.action import ReachGoal

from vjepa_byor.dashboard import Commands, Hub
from vjepa_byor.ros2.config import CameraConfig, StateConfig
from vjepa_byor.ros2.link import image_to_tensor, qos_profile, read_state, stamp_ns
from vjepa_byor.ros2.reach_goal import ACTION_NAME, LATCHED

MAX_CAMERA_HZ = 10.0


def data_url(msg: Image) -> str:
    """A ``sensor_msgs/Image`` as a PNG data URL, at its own size."""
    photo = image_to_tensor(msg, (msg.height, msg.width))
    buffer = io.BytesIO()
    PILImage.fromarray(photo.permute(1, 2, 0).numpy()).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class DashboardNode(Node):
    def __init__(
        self,
        hub: Hub,
        camera: CameraConfig,
        state: StateConfig,
        action_name: str = ACTION_NAME,
        reset_service: str | None = None,
        **node_kwargs: Any,
    ) -> None:
        super().__init__("vjepa_dashboard", **node_kwargs)
        self.hub, self.state_config = hub, state
        self._camera_ns, self._camera_at = -1, 0.0
        self._last_state: list[float] | None = None
        state_cls = _message_class(state.type)
        self.create_subscription(Image, camera.topic, self._on_camera, qos_profile(camera.qos))
        self.create_subscription(state_cls, state.topic, self._on_state, qos_profile(state.qos))
        self.create_subscription(Image, f"{action_name}/goal_image", self._on_goal_image, LATCHED)
        self.create_subscription(String, f"{action_name}/events", self._on_event, 50)
        self.client = ActionClient(self, ReachGoal, action_name)
        self.cancel_client = self.create_client(CancelGoal, f"{action_name}/_action/cancel_goal")
        self.reset_client = self.create_client(Trigger, reset_service) if reset_service else None
        self.next_client = self.create_client(Trigger, f"{action_name}/next")
        self.action_name = action_name

    @property
    def commands(self) -> Commands:
        return Commands(send_goal=self.send_goal, cancel=self.cancel, next_step=self.next_step,
                        reset=self.reset if self.reset_client is not None else None)

    # Incoming

    def _on_camera(self, msg: Image) -> None:
        ns, now = stamp_ns(msg.header.stamp), time.monotonic()
        if ns == self._camera_ns or now - self._camera_at < 1.0 / MAX_CAMERA_HZ:
            return  # a repeat of the same view, or faster than the page needs
        self._camera_ns, self._camera_at = ns, now
        try:
            self.hub.publish({"type": "camera", "src": data_url(msg)})
        except ValueError as e:
            self.get_logger().warning(f"camera: {e}", throttle_duration_sec=10.0)

    def _on_goal_image(self, msg: Image) -> None:
        self.hub.publish({"type": "goal_image", "src": data_url(msg)})

    def _on_state(self, msg: Any) -> None:
        try:
            values = read_state(self.state_config, msg)
        except (KeyError, IndexError, AttributeError):
            return
        if values != self._last_state:
            self._last_state = values
            names = [f.name for f in self.state_config.fields]
            self.hub.publish({"type": "state", "state": dict(zip(names, values))})

    def _on_event(self, msg: String) -> None:
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(event, dict) and event.get("type") in ("start", "pending", "step", "end"):
            self.hub.publish(event)

    # Commands from the page (HTTP threads)

    def send_goal(self, state: list[float], step_by_step: bool = False) -> str:
        if self.hub.running:
            raise RuntimeError("a goal is already running")
        if not self.client.server_is_ready():
            raise RuntimeError(f"no {self.action_name} server: start apps/ros2/reach_goal_server.py "
                               "(same ROS_DOMAIN_ID)")
        future = self.client.send_goal_async(ReachGoal.Goal(goal_state=state, step_by_step=step_by_step))
        future.add_done_callback(self._on_goal_response)
        mode = ", step by step" if step_by_step else ""
        return f"goal sent: ({', '.join(f'{v:g}' for v in state)}){mode}"

    def next_step(self) -> str:
        if not self.next_client.service_is_ready():
            raise RuntimeError(f"no {self.action_name} server")
        future = self.next_client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: None if f.result() is not None and f.result().success
            else self.hub.notice(f.result().message if f.result() is not None else "next failed", "error"))
        return ""

    def _on_goal_response(self, future) -> None:
        handle = future.result()
        if handle is None or not handle.accepted:
            self.hub.notice("goal rejected: another goal is running", "error")

    def cancel(self) -> str:
        if not self.cancel_client.service_is_ready():
            raise RuntimeError(f"no {self.action_name} server")
        self.cancel_client.call_async(CancelGoal.Request())  # zero goal id and stamp: cancel every goal
        return "cancelling"

    def reset(self) -> str:
        if not self.reset_client.service_is_ready():
            raise RuntimeError(f"no {self.reset_client.srv_name} service: is the robot running "
                               "(python -m vjepa_byor_sim.ros2_bridge), on the same ROS_DOMAIN_ID?")
        future = self.reset_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_reset)
        return "resetting"

    def _on_reset(self, future) -> None:
        response = future.result()
        if response is None or not response.success:
            self.hub.notice(response.message if response is not None else "reset failed", "error")
            return
        self.hub.clear()
        self.hub.notice(response.message)


def _message_class(name: str):
    from rosidl_runtime_py.utilities import get_message

    return get_message(name)
