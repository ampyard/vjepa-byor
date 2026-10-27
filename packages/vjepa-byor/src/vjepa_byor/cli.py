"""Command line: ``vjepa-byor collect`` (``encode``, ``finetune`` and ``eval`` come with module 4)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from vjepa_byor.data.collector import CollectionSummary, CollectorConfig, CollectRun, collect
from vjepa_byor.data.dataset import verify
from vjepa_byor.data.policy import EdgeAwareRandomWalk
from vjepa_byor.data.storage import DatasetFormatError
from vjepa_byor.registry import RigNotFoundError, load_rig


def _load_rig_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise SystemExit(f"{path}: reading YAML needs PyYAML (pip install pyyaml), or use a .json file") from None
        options = yaml.safe_load(text)
    else:
        options = json.loads(text)
    if not isinstance(options, dict):
        raise SystemExit(f"{path}: the rig config must be a mapping of option names to values")
    return options


def _parse_bounds(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``"0:200,0:150"`` -> (low, high) on the state tensor."""
    try:
        pairs = [tuple(float(v) for v in part.split(":")) for part in text.split(",")]
        low, high = zip(*pairs, strict=True)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected LOW:HIGH[,LOW:HIGH...], got {text!r}") from None
    return torch.tensor(low), torch.tensor(high)


def _add_collect(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "collect",
        help="record (before photo, state, action, after photo) samples from a rig",
        description="Drive a rig through random episodes and record every transition.",
    )
    p.add_argument("--rig", help="registered rig name, e.g. sim or esp32-stepper")
    p.add_argument("--out", type=Path, help="dataset directory")
    p.add_argument("--num-samples", type=int, default=1000, help="target total, including earlier runs (default 1000)")
    p.add_argument("--steps-per-episode", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settle-time", type=float, default=0.5, help="seconds between a move and its photo (default 0.5)")
    p.add_argument("--reply-timeout", type=float, help="seconds to wait for the rig's reply (default: the rig's own)")
    p.add_argument("--camera-retries", type=int, default=2)
    p.add_argument("--state-bounds", type=_parse_bounds, help="limits on the state tensor, LOW:HIGH per dimension")
    p.add_argument(
        "--reject-rate", type=float, default=0.02, help="share of deliberately out-of-range moves (default 0.02)"
    )
    p.add_argument("--rig-config", type=Path, help="JSON (or YAML, with PyYAML) options for the rig's factory")
    p.add_argument("--resume", action="store_true", help="add to an existing dataset in --out")
    p.add_argument("--verify", type=Path, metavar="ROOT", help="check a dataset and exit")
    p.set_defaults(handler=_collect)


def _collect(args: argparse.Namespace) -> int:
    if args.verify is not None:
        report = verify(args.verify)
        print(report.format())
        return 0 if report.ok else 1
    if not args.rig or args.out is None:
        print("collect: --rig and --out are required (or use --verify ROOT)", file=sys.stderr)
        return 2

    options = _load_rig_config(args.rig_config) if args.rig_config else {}
    try:
        rig = load_rig(args.rig, **options)
    except RigNotFoundError as e:
        print(f"collect: {e}", file=sys.stderr)
        return 2

    interactive = sys.stdout.isatty()
    last: list[CollectionSummary] = []

    def progress(summary: CollectionSummary) -> None:
        last[:] = [summary]
        if interactive:
            print("\r" + summary.format(), end="", flush=True)
        elif summary.samples % 50 == 0:
            print(summary.format(), flush=True)

    run = CollectRun(
        config=CollectorConfig(
            num_samples=args.num_samples,
            steps_per_episode=args.steps_per_episode,
            seed=args.seed,
            settle_time=args.settle_time,
            reply_timeout=args.reply_timeout,
            camera_retries=args.camera_retries,
        ),
        policy=EdgeAwareRandomWalk(rig.action_space.actions(), bounds=args.state_bounds, reject_rate=args.reject_rate),
        resume=args.resume,
        rig_options=options,
        on_progress=progress,
    )
    try:
        summary = collect(rig, args.out, run)
    except (FileExistsError, DatasetFormatError, ValueError) as e:
        print(f"collect: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\naborted; the step in flight was discarded")
        if last:
            print(last[0].format())
        return 130
    if interactive:
        print()
    print(summary.format())
    if summary.stopped:
        print(f"stopped early: {summary.stopped}")
    print(f"dataset: {args.out}  (check it with: vjepa-byor collect --verify {args.out})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vjepa-byor", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_collect(subparsers)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
