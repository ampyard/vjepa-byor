"""Runs the planner on the simulated rig: one goal, or a batch of random goals for evaluation.

    # one run, printing every step
    uv run python apps/planner/sim_planner.py run --start 40 100 --goal 120 60

    # 40 random goals 10-100 mm away; summary by distance, per-run JSON
    uv run python apps/planner/sim_planner.py batch --goals 40 --json runs.json

    # 2-step look-ahead, second step tries only the 20 mm moves
    uv run python apps/planner/sim_planner.py batch --horizon 2 --lookahead-sizes 20

The goal photo is rendered by the sim at the goal position, so no goal image
generator is needed. POC-specific: lives in apps/, not in the library.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from vjepa_byor.planning import Planner, PlannerConfig, PlanStep, make_scorer
from vjepa_byor.predictor import ActionPredictor, load_finetuned, load_pretrained
from vjepa_byor_sim import DEFAULT_TRAVEL_MM, SimRig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "sim-10k"
REACHED_MM = 5.0


def load_model(finetuned: Path, device: str) -> ActionPredictor:
    dev = torch.device(device)
    encoder = load_pretrained(predictor=False).encoder.to(dev)
    predictor = load_finetuned(finetuned).to(dev)
    return ActionPredictor(encoder, predictor, SimRig().mapping)


def config_from(args: argparse.Namespace) -> PlannerConfig:
    lookahead = None
    if args.lookahead_sizes:
        labels = SimRig().action_space.labels()
        lookahead = [i for i, label in enumerate(labels) if int(label.split()[1].removesuffix("mm")) in args.lookahead_sizes]
    return PlannerConfig(threshold=args.threshold, max_steps=args.max_steps, horizon=args.horizon,
                         lookahead_actions=lookahead, beam=args.beam)


def plan_once(model: ActionPredictor, config: PlannerConfig, scorer, start, goal, on_step=None) -> dict:
    rig = SimRig(start_mm=start)
    goal_photo = SimRig(start_mm=goal).camera.capture()
    with rig:
        result = Planner(model, rig, config, scorer).run(goal_photo, rig.transport.responder.state, on_step=on_step)
    final = (result.final_state["x_mm"], result.final_state["y_mm"])
    return {
        **result.to_dict(),
        "start": list(start), "goal": list(goal), "distance_mm": math.dist(start, goal),
        "final_distance_mm": math.dist(final, goal), "reached": math.dist(final, goal) <= REACHED_MM,
        "planner_says_reached": result.reached,
    }


def random_goals(n: int, seed: int, low: float, high: float) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Start/goal pairs ``low``-``high`` mm apart, inside the travel area. Seed 0 matches the CFP batch."""
    rnd = random.Random(seed)
    w, h = DEFAULT_TRAVEL_MM
    pairs = []
    for _ in range(n):
        while True:
            s = (rnd.uniform(10, w - 10), rnd.uniform(10, h - 10))
            d, a = rnd.uniform(low, high), rnd.uniform(0, 2 * math.pi)
            g = (s[0] + d * math.cos(a), s[1] + d * math.sin(a))
            if 5 <= g[0] <= w - 5 and 5 <= g[1] <= h - 5:
                break
        pairs.append((tuple(round(v) for v in s), tuple(round(v) for v in g)))
    return pairs


def print_step(step: PlanStep) -> None:
    pos = f"({step.state['x_mm']:6.1f}, {step.state['y_mm']:6.1f})"
    move = f"-> {step.chosen:8s} predicted {step.chosen_error:.4f}" if step.chosen else ""
    refused = f"  refused {step.refused}" if step.refused else ""
    print(f"step {step.step:2d} at {pos}  error {step.current_error:.4f}  {move}  [{step.seconds:.2f} s]{refused}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["run", "batch"])
    p.add_argument("--finetuned", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--scorer", default="l1")
    p.add_argument("--threshold", type=float, default=0.165)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--lookahead-sizes", type=int, nargs="*", help="step sizes (mm) tried after the first step")
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--start", type=float, nargs=2, default=(40, 100))
    p.add_argument("--goal", type=float, nargs=2, default=(120, 60))
    p.add_argument("--goals", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-mm", type=float, default=10)
    p.add_argument("--max-mm", type=float, default=100)
    p.add_argument("--json", type=Path, help="write the run(s) here")
    args = p.parse_args()

    model = load_model(args.finetuned, args.device)
    config, scorer = config_from(args), make_scorer(args.scorer)

    if args.mode == "run":
        run = plan_once(model, config, scorer, tuple(args.start), tuple(args.goal), on_step=print_step)
        print(f"{run['reason']}: {run['moves']} moves, {run['final_distance_mm']:.1f} mm from the goal")
        runs = [run]
    else:
        runs, t0 = [], time.perf_counter()
        for i, (s, g) in enumerate(random_goals(args.goals, args.seed, args.min_mm, args.max_mm)):
            run = plan_once(model, config, scorer, s, g)
            runs.append(run)
            print(f"{i:3d} {run['distance_mm']:5.1f} mm -> {run['final_distance_mm']:5.1f} mm  "
                  f"{run['moves']:2d} moves  {run['reason']}")
        print(f"\n{config}\n{len(runs)} runs in {time.perf_counter() - t0:.0f} s")
        for lo, hi in ((0, 60), (60, math.inf), (0, math.inf)):
            b = [r for r in runs if lo <= r["distance_mm"] < hi]
            ok = [r for r in b if r["reached"]]
            if b:
                steps = [s["seconds"] for r in b for s in r["steps"]]
                print(f"  goals {lo:>3}-{hi:<5} mm: reached {len(ok):2d}/{len(b):<2d}  "
                      f"mean moves (reached) {sum(r['moves'] for r in ok) / max(1, len(ok)):4.1f}  "
                      f"self-stopped {sum(r['reason'] == 'reached' for r in b):2d}  "
                      f"mean decision {sum(steps) / len(steps):.2f} s")
    if args.json:
        args.json.write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
