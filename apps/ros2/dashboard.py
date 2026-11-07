"""The demo dashboard: a web page showing the robot and every /reach_goal run (demo backlog item 7).

In the ROS 2 container, next to the bridge and the /reach_goal server:

    python apps/ros2/dashboard.py --host ryzen.tail123187.ts.net    # then open http://<that name>:8080

It listens only on the address ``--host`` resolves to (one IPv4 address), so only that interface
serves it. Use the full Tailscale MagicDNS name: the short host name usually resolves to 127.0.1.1
(/etc/hosts), which only this machine can reach.

Topics come from the same ros2 rig config as the server. ``--mat sim`` (the default) tells the page
where the sim's mat is in the photo, so a click on it becomes a goal in millimetres. Every run is
saved to ``--runs-dir`` and can be replayed from the page.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
from pathlib import Path

import rclpy
import yaml
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

from vjepa_byor.dashboard import DashboardServer, Hub
from vjepa_byor.dashboard.web import resolve_host
from vjepa_byor.ros2.config import CameraConfig, StateConfig
from vjepa_byor.ros2.dashboard import DashboardNode

ROOT = Path(__file__).resolve().parents[2]


def sim_mat(resolution) -> dict:
    """Where the sim's travel area is in its photo: ``px = x0 + x_mm * px_per_mm``, ``py = y0 - y_mm * px_per_mm``."""
    from vjepa_byor_sim import SimCamera, SimStage

    stage = SimStage()
    camera = SimCamera(stage, resolution)
    x0, y0 = camera.to_pixel(0, 0)
    return {"x0": x0, "y0": y0, "px_per_mm": camera.px_per_mm, "travel": [stage.width_mm, stage.height_mm]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig-config", type=Path, default=ROOT / "apps" / "ros2" / "sim_robot.yaml")
    p.add_argument("--host", required=True,
                   help="name or address to listen on, e.g. your Tailscale MagicDNS name; only that interface serves")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--runs-dir", type=Path, default=ROOT / "data" / "dashboard-runs")
    p.add_argument("--mat", choices=["sim", "none"], default="sim", help="click-to-goal geometry")
    p.add_argument("--reset-service", default="/rig/reset", help="std_srvs/Trigger service, or '' for none")
    args, ros_args = p.parse_known_args(argv)

    try:
        address = resolve_host(args.host)
    except ValueError as e:
        p.error(str(e))
    if ipaddress.ip_address(address).is_loopback and args.host != "localhost":
        print(f"warning: {args.host} is {address}, a loopback address: only this machine can open the page. "
              "For Tailscale use the full MagicDNS name (name.tailnet.ts.net).", file=sys.stderr)

    rig_config = yaml.safe_load(args.rig_config.read_text())
    camera, state = CameraConfig.parse(rig_config["camera"]), StateConfig.parse(rig_config["state"])

    rclpy.init(args=ros_args)
    hub = Hub(args.runs_dir)
    node = DashboardNode(hub, camera, state, reset_service=args.reset_service or None)
    config = {
        "state_names": [f.name for f in state.fields],
        "camera": list(camera.resolution),
        "mat": sim_mat(camera.resolution) if args.mat == "sim" else None,
        "can_goal": True,
        "can_reset": bool(args.reset_service),
    }
    web = DashboardServer(hub, node.commands, config, args.host, args.port)
    web.start()
    node.get_logger().info(
        f"dashboard on http://{args.host}:{web.port} (listening on {web.address} only; runs saved to {args.runs_dir})")
    # single-threaded: a MultiThreadedExecutor spins at ~100% CPU with the camera subscription
    # (rclpy, Jazzy), which made the page and its buttons lag; the callbacks here are all short
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        web.stop()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
