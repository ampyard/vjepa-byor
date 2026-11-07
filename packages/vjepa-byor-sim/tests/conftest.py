"""Helpers for the ROS 2 tests (they skip themselves without rclpy)."""

import threading

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
except ImportError:  # host without ROS
    rclpy = None


class BridgeThread:
    """The sim bridge node in its own context, spinning on a background thread."""

    def __init__(self, domain_id, start_mm=(100.0, 75.0), rate_hz=10.0):
        from vjepa_byor_sim.ros2_bridge import SimBridge

        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=domain_id)
        self.node = SimBridge(
            context=self.context,
            parameter_overrides=[Parameter("start_mm", value=list(start_mm)), Parameter("rate_hz", value=rate_hz)],
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.spin, daemon=True)
        self.thread.start()

    def spin(self):
        while not self.stop.is_set():
            self.executor.spin_once(timeout_sec=0.05)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.try_shutdown(context=self.context)
