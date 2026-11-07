"""ROS 2 bridge node: the simulated stage as an ordinary ROS 2 robot (ROSCon demo backlog item 4).

    python -m vjepa_byor_sim.ros2_bridge                                  # defaults
    python -m vjepa_byor_sim.ros2_bridge --ros-args -p start_mm:="[40.0, 100.0]" -p rate_hz:=5.0

Topics (SI units, as ROS expects):

- ``/camera/image_raw`` (``sensor_msgs/Image``, ``rgb8``, 256 x 256): the top-down camera
- ``/rig/state`` (``sensor_msgs/JointState``): prismatic joints ``x`` and ``y``, positions in metres
- ``/rig/cmd`` (``geometry_msgs/Vector3Stamped``, subscribed): a relative move, ``vector.x`` and
  ``vector.y`` in metres; ``z`` is ignored
- ``/rig/reset`` (``std_srvs/Trigger``, service): puts the carriage back at ``start_mm``

The photo and the state are published together with the same ``header.stamp``, so they can be
matched. They go out right after every command, *including a refused one* (a move that would leave
the travel area), with a new stamp: a state stamped after the command means the move has finished,
and an unchanged position means it was refused. Moves are instant. In between they are repeated at
``rate_hz`` with the stamp of the last command (or reset): the view has not changed since, and a
fresh stamp there could pass for the answer to a command the bridge has not handled yet.

Needs ``rclpy`` from a sourced ROS 2 environment (see ``docker/README.md``); nothing else in this
package imports it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import rclpy
import rclpy.executors
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
from torch import Tensor

from vjepa_byor_sim.camera import DEFAULT_RESOLUTION, SimCamera
from vjepa_byor_sim.stage import DEFAULT_TRAVEL_MM, SimStage

NODE_NAME = "sim_bridge"
JOINT_NAMES = ("x", "y")
MM_PER_M = 1000.0

# Latest photo and state stay available to late subscribers (e.g. the planner starting up)
LATCHED = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def image_msg(photo: Tensor, stamp: Time, frame_id: str) -> Image:
    """A ``(3, H, W)`` ``uint8`` photo as an ``rgb8`` ``sensor_msgs/Image`` (no cv_bridge needed)."""
    _, height, width = photo.shape
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.step = height, width, "rgb8", 0, 3 * width
    msg.data = photo.permute(1, 2, 0).contiguous().numpy().tobytes()
    return msg


def joint_state_msg(x_mm: float, y_mm: float, stamp: Time, frame_id: str) -> JointState:
    msg = JointState()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.name = list(JOINT_NAMES)
    msg.position = [x_mm / MM_PER_M, y_mm / MM_PER_M]
    return msg


class SimBridge(Node):
    """Publishes the sim's camera and state, and moves the carriage on ``/rig/cmd``."""

    def __init__(self, **kwargs) -> None:
        super().__init__(NODE_NAME, **kwargs)
        travel = self.declare_parameter("travel_mm", list(DEFAULT_TRAVEL_MM)).value
        start = self.declare_parameter("start_mm", [travel[0] / 2, travel[1] / 2]).value
        resolution = self.declare_parameter("resolution", list(DEFAULT_RESOLUTION)).value
        rate_hz = self.declare_parameter("rate_hz", 10.0).value
        self.camera_frame = self.declare_parameter("camera_frame", "camera").value
        self.stage_frame = self.declare_parameter("stage_frame", "stage").value

        self.start_mm = (float(start[0]), float(start[1]))
        self.stage = SimStage(travel, self.start_mm)
        self.camera = SimCamera(self.stage, resolution)
        self._photo: Tensor | None = None
        self._photo_at: tuple[float, float] | None = None
        self._stamp: Time | None = None  # when the current state was last set or confirmed

        self.image_pub = self.create_publisher(Image, "/camera/image_raw", LATCHED)
        self.state_pub = self.create_publisher(JointState, "/rig/state", LATCHED)
        self.create_subscription(Vector3Stamped, "/rig/cmd", self.on_cmd, 10)
        self.create_service(Trigger, "/rig/reset", self.on_reset)
        if rate_hz > 0:
            self.create_timer(1.0 / rate_hz, self.republish)

        self.get_logger().info(
            f"sim stage {travel[0]:g} x {travel[1]:g} mm, carriage at {self._where()}, "
            f"camera {self.camera.resolution[1]} x {self.camera.resolution[0]}, {rate_hz:g} Hz"
        )
        self.publish()

    def _where(self) -> str:
        return f"({self.stage.x_mm:.1f}, {self.stage.y_mm:.1f}) mm"

    def photo(self) -> Tensor:
        """The current photo; rendered again only when the carriage has moved."""
        at = (self.stage.x_mm, self.stage.y_mm)
        if at != self._photo_at:
            self._photo, self._photo_at = self.camera.capture(), at
        return self._photo

    def publish(self) -> None:
        """Publishes the current photo and state with a new stamp (after a command, a reset or at start)."""
        self._stamp = self.get_clock().now().to_msg()
        self._send(self._stamp)

    def republish(self) -> None:
        """The periodic repeat: same view, same stamp."""
        self._send(self._stamp)

    def _send(self, stamp: Time) -> None:
        self.image_pub.publish(image_msg(self.photo(), stamp, self.camera_frame))
        self.state_pub.publish(joint_state_msg(self.stage.x_mm, self.stage.y_mm, stamp, self.stage_frame))

    def move(self, dx_mm: float, dy_mm: float) -> bool:
        """Moves the carriage; ``False`` (and no motion) if the move would leave the travel area."""
        x, y = self.stage.x_mm + dx_mm, self.stage.y_mm + dy_mm
        if not self.stage.in_range(x, y):
            return False
        self.stage.x_mm, self.stage.y_mm = x, y
        return True

    def on_cmd(self, msg: Vector3Stamped) -> None:
        dx, dy = msg.vector.x * MM_PER_M, msg.vector.y * MM_PER_M
        if not (math.isfinite(dx) and math.isfinite(dy)):
            self.get_logger().warning(f"ignored a non-finite move {[msg.vector.x, msg.vector.y]}")
        elif self.move(dx, dy):
            self.get_logger().info(f"moved ({dx:+.1f}, {dy:+.1f}) mm -> {self._where()}")
        else:
            self.get_logger().warning(f"refused ({dx:+.1f}, {dy:+.1f}) mm: out of range, still at {self._where()}")
        self.publish()  # even when refused: tells the sender the command has been handled

    def on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.stage.x_mm, self.stage.y_mm = self.start_mm
        self.publish()
        response.success, response.message = True, f"carriage back at {self._where()}"
        return response


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
