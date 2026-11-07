"""The ``ros2`` rig's connection to ROS 2: one node, shared by the transport and the camera.

Imports ``rclpy`` only in :meth:`Ros2Link.open`, so the rest of the package works without ROS.
The node runs in its own ``rclpy`` context with an executor on a background thread; callbacks
store the newest state and photo, and :meth:`Ros2ActionTransport.execute` and
:meth:`Ros2Camera.capture` wait on them.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from vjepa_byor.interfaces import RigError, RigState
from vjepa_byor.ros2.config import CameraConfig, CommandConfig, StateConfig

NODE_NAME = "vjepa_byor_rig"
_CHANNELS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}


def stamp_ns(stamp: Any) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def get_path(msg: Any, path: str) -> Any:
    for part in path.split("."):
        msg = getattr(msg, part)
    return msg


def set_path(msg: Any, path: str, value: float) -> None:
    *parents, last = path.split(".")
    for part in parents:
        msg = getattr(msg, part)
    setattr(msg, last, type(getattr(msg, last))(value))


def image_to_tensor(msg: Any, resolution: tuple[int, int]) -> torch.Tensor:
    """A ``sensor_msgs/Image`` as an RGB ``uint8`` tensor ``(3, H, W)``, resized to ``resolution``."""
    channels = _CHANNELS.get(msg.encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}; use one of {', '.join(_CHANNELS)}")
    rows = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8).view(msg.height, msg.step)
    hwc = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding in ("bgr8", "bgra8"):
        hwc = hwc[..., [2, 1, 0]]
    elif msg.encoding == "mono8":
        hwc = hwc.expand(-1, -1, 3)
    image = hwc[..., :3].permute(2, 0, 1).contiguous()
    if tuple(image.shape[1:]) != tuple(resolution):
        resized = F.interpolate(image[None].float(), size=resolution, mode="bilinear", antialias=True)
        image = resized[0].round().clamp(0, 255).to(torch.uint8)
    return image


def tensor_to_image(photo: torch.Tensor, stamp: Any, frame_id: str = "") -> Any:
    """A ``(3, H, W)`` ``uint8`` photo as an ``rgb8`` ``sensor_msgs/Image``."""
    from sensor_msgs.msg import Image

    _, height, width = photo.shape
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.step = height, width, "rgb8", 0, 3 * width
    msg.data = photo.permute(1, 2, 0).contiguous().numpy().tobytes()
    return msg


def _qos(name: str):
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    reliability = ReliabilityPolicy.BEST_EFFORT if name == "best_effort" else ReliabilityPolicy.RELIABLE
    durability = DurabilityPolicy.TRANSIENT_LOCAL if name == "latched" else DurabilityPolicy.VOLATILE
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=reliability, durability=durability)


class Ros2Link:
    """The node, its subscriptions and the command publisher; opened by the transport (or the camera)."""

    def __init__(
        self,
        camera: CameraConfig,
        state: StateConfig,
        command: CommandConfig,
        state_type: type[RigState],
        *,
        domain_id: int | None = None,
        connect_timeout: float = 10.0,
    ) -> None:
        self.camera, self.state, self.command = camera, state, command
        self.state_type = state_type
        self.domain_id, self.connect_timeout = domain_id, connect_timeout
        self.cond = threading.Condition()
        self._context = self._node = self._executor = self._thread = None
        self._reset_data()

    def _reset_data(self) -> None:
        self.state_values: list[float] | None = None
        self.state_ns = 0  # header.stamp of the newest state (receive time for messages without a header)
        self.image_msg: Any = None
        self.image_ns = 0
        self.last_move_ns = 0  # stamp of the state the last execute() returned

    @property
    def is_open(self) -> bool:
        return self._node is not None

    def open(self) -> None:
        if self.is_open:
            return
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.signals import SignalHandlerOptions
            from rosidl_runtime_py.utilities import get_message
            from sensor_msgs.msg import Image
        except ImportError as e:
            raise ConnectionError(
                f"the ros2 rig needs rclpy from a sourced ROS 2 environment ({e}); see docker/README.md"
            ) from e

        try:
            state_cls, self._cmd_cls = get_message(self.state.type), get_message(self.command.type)
        except (AttributeError, ModuleNotFoundError, ValueError) as e:
            raise ConnectionError(f"unknown ROS 2 message type: {e}") from e
        for f in self.command.fields:
            self._check_path(self._cmd_cls(), f.field, "command")
        self._check_state_paths(state_cls())

        self._reset_data()
        self._context = rclpy.Context()
        # no signal handlers: Ctrl-C stays with the caller (the collector stops cleanly on it)
        rclpy.init(context=self._context, domain_id=self.domain_id, signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node(NODE_NAME, context=self._context)
        self._node.create_subscription(state_cls, self.state.topic, self._on_state, _qos(self.state.qos))
        self._node.create_subscription(Image, self.camera.topic, self._on_image, _qos(self.camera.qos))
        self._pub = self._node.create_publisher(self._cmd_cls, self.command.topic, 10)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, name="vjepa-byor-ros2", daemon=True)
        self._thread.start()

        end = time.monotonic() + self.connect_timeout
        while self._pub.get_subscription_count() == 0 or self.state_values is None:
            if time.monotonic() > end:
                waiting = []
                if self._pub.get_subscription_count() == 0:
                    waiting.append(f"a subscriber on {self.command.topic}")
                if self.state_values is None:
                    waiting.append(f"a state on {self.state.topic}")
                self.close()
                raise ConnectionError(
                    f"no robot after {self.connect_timeout:g} s: still waiting for {' and '.join(waiting)} "
                    "(is the robot running, on the same ROS_DOMAIN_ID?)"
                )
            time.sleep(0.02)

    def _check_path(self, msg: Any, path: str, where: str) -> None:
        try:
            value = get_path(msg, path)
        except AttributeError:
            raise ConnectionError(f"{where}: {type(msg).__name__} has no field {path!r}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConnectionError(f"{where}: {type(msg).__name__}.{path} is not a number")

    def _check_state_paths(self, msg: Any) -> None:
        for f in self.state.fields:
            if f.field is not None:
                self._check_path(msg, f.field, "state")

    def _spin(self) -> None:
        while not self._stop.is_set() and self._context.ok():
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:  # shutting down
                if self._stop.is_set():
                    return
                raise

    def close(self) -> None:
        if not self.is_open:
            return
        import rclpy

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._executor.shutdown()
        self._node.destroy_node()
        rclpy.try_shutdown(context=self._context)
        self._context = self._node = self._executor = self._thread = None
        with self.cond:
            self.cond.notify_all()

    def require_open(self) -> None:
        if not self.is_open:
            raise ConnectionError("the ros2 rig is not open")

    # Callbacks (executor thread)

    def _msg_ns(self, msg: Any) -> int:
        header = getattr(msg, "header", None)
        return stamp_ns(header.stamp) if header is not None else self._node.get_clock().now().nanoseconds

    def _on_state(self, msg: Any) -> None:
        try:
            values = self._read_state(msg)
        except (KeyError, IndexError, AttributeError) as e:
            self._node.get_logger().warning(f"ignored a state on {self.state.topic}: {e!r}")
            return
        with self.cond:
            self.state_values, self.state_ns = values, self._msg_ns(msg)
            self.cond.notify_all()

    def _read_state(self, msg: Any) -> list[float]:
        values = []
        for f in self.state.fields:
            if f.joint is not None:
                raw = msg.position[list(msg.name).index(f.joint)] if f.joint in msg.name else None
                if raw is None:
                    raise KeyError(f"no joint {f.joint!r} in {list(msg.name)}")
            else:
                raw = get_path(msg, f.field)
            values.append(float(raw) * f.scale)
        return values

    def _on_image(self, msg: Any) -> None:
        with self.cond:
            self.image_msg, self.image_ns = msg, self._msg_ns(msg)
            self.cond.notify_all()

    # Used by the transport and the camera

    def wait(self, ready: Callable[[], bool], timeout: float | None) -> bool:
        """Waits (holding :attr:`cond`) until ``ready()`` or the link closes; ``False`` on timeout."""
        return self.cond.wait_for(lambda: ready() or not self.is_open, timeout) and self.is_open

    def now_ns(self) -> int:
        return self._node.get_clock().now().nanoseconds

    def publish_command(self, action: torch.Tensor, stamp: int) -> None:
        from builtin_interfaces.msg import Time

        msg = self._cmd_cls()
        if hasattr(msg, "header"):
            msg.header.stamp = Time(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000)
        for value, f in zip(action.tolist(), self.command.fields, strict=True):
            set_path(msg, f.field, value * f.scale)
        self._pub.publish(msg)


class Ros2ActionTransport:
    """:class:`~vjepa_byor.interfaces.ActionTransport` over ROS 2: publish a command, wait for the next state."""

    def __init__(self, link: Ros2Link) -> None:
        self.link = link

    @property
    def is_open(self) -> bool:
        return self.link.is_open

    def open(self) -> None:
        self.link.open()

    def close(self) -> None:
        self.link.close()

    def current_state(self) -> RigState:
        """The newest state the robot published, without moving (the planner's start state)."""
        self.link.require_open()
        with self.link.cond:
            return self.link.state_type(*self.link.state_values)

    def execute(self, action: torch.Tensor, timeout: float | None = None) -> RigState:
        link, command = self.link, self.link.command
        link.require_open()
        timeout = command.timeout if timeout is None else timeout
        with link.cond:
            before = link.state_values
            sent = link.now_ns()
            link.publish_command(action, sent)
            if not link.wait(lambda: link.state_ns > sent, timeout):
                link.require_open()
                raise TimeoutError(f"no state on {link.state.topic} within {timeout:g} s of the command")
            after, link.last_move_ns = link.state_values, link.state_ns
        state = link.state_type(*after)
        moved = any(abs(a - b) > command.tolerance for a, b in zip(after, before))
        if command.refused_if_unchanged and not moved and bool(action.abs().sum() > 0):
            raise RigError(f"move refused: the state stayed at {state.to_dict()}")
        return state


class Ros2Camera:
    """:class:`~vjepa_byor.interfaces.Camera` reading ``sensor_msgs/Image``: the first photo no older than the last move."""

    def __init__(self, link: Ros2Link) -> None:
        self.link = link

    @property
    def resolution(self) -> tuple[int, int]:
        return self.link.camera.resolution

    def capture(self) -> torch.Tensor:
        link = self.link
        link.open()  # no-op when the transport already opened it
        with link.cond:
            if not link.wait(lambda: link.image_msg is not None and link.image_ns >= link.last_move_ns,
                             link.camera.timeout):
                link.require_open()
                raise TimeoutError(f"no photo on {link.camera.topic} within {link.camera.timeout:g} s")
            msg = link.image_msg
        return image_to_tensor(msg, self.resolution)
