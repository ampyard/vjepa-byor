"""The planner as a ROS 2 action server, ``/reach_goal``, on the robot described by a ros2 rig config.

In the ROS 2 container (docker/run.sh), with the sim bridge running in another terminal:

    python -m vjepa_byor_sim.ros2_bridge
    python apps/ros2/reach_goal_server.py
    ros2 action send_goal --feedback /reach_goal vjepa_byor_interfaces/action/ReachGoal "{goal_state: [120, 60]}"

``--goal-renderer sim`` (the default) lets goals be given as a rig state: the sim renders the goal
photo at that position. With ``none``, goals must be a photo (``goal_image``).
POC-specific (model loading, the sim renderer): lives in apps/, the server is vjepa_byor.ros2.reach_goal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rclpy
import torch
import yaml
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from vjepa_byor import load_rig
from vjepa_byor.planning import PlannerConfig, make_scorer
from vjepa_byor.predictor import ActionPredictor, load_finetuned, load_pretrained
from vjepa_byor.ros2.reach_goal import ReachGoalServer

ROOT = Path(__file__).resolve().parents[2]


def sim_renderer(resolution):
    from vjepa_byor_sim import SimRig

    def render(state):
        return SimRig(start_mm=state, resolution=resolution).camera.capture()

    return render


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig-config", type=Path, default=ROOT / "apps" / "ros2" / "sim_robot.yaml")
    p.add_argument("--finetuned", type=Path, default=ROOT / "checkpoints" / "sim-10k")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--scorer", default="l1")
    p.add_argument("--threshold", type=float, default=0.165)
    p.add_argument("--max-steps", type=int, default=20, help="default for goals that don't set max_steps")
    p.add_argument("--goal-renderer", choices=["sim", "none"], default="sim")
    args, ros_args = p.parse_known_args(argv)

    rig = load_rig("ros2", **yaml.safe_load(args.rig_config.read_text()))
    print(f"loading the model on {args.device} ...", file=sys.stderr)
    device = torch.device(args.device)
    encoder = load_pretrained(predictor=False).encoder.to(device)
    model = ActionPredictor(encoder, load_finetuned(args.finetuned).to(device), rig.mapping)
    render = sim_renderer(rig.camera.resolution) if args.goal_renderer == "sim" else None

    rclpy.init(args=ros_args)
    server = ReachGoalServer(
        model, rig, PlannerConfig(threshold=args.threshold, max_steps=args.max_steps), make_scorer(args.scorer), render
    )
    try:
        rig.open()
    except ConnectionError as e:
        server.get_logger().warning(f"{e}; trying again on the first goal")
    executor = MultiThreadedExecutor()
    executor.add_node(server)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rig.close()
        server.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
