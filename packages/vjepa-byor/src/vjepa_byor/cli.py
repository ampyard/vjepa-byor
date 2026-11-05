"""Command line: ``vjepa-byor collect | encode | finetune | eval``."""

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
from vjepa_byor.data.storage import DatasetFormatError, read_meta
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


# encode | finetune | eval (the predictor modules import torch.hub code lazily, so collect stays light)


def _device(name: str | None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def _model_id(checkpoint: Path | None) -> str:
    from vjepa_byor.predictor import loading

    return f"{loading.MODEL_ID}:{(checkpoint or loading.default_checkpoint()).name}"


def _rig_mapping(args: argparse.Namespace, dataset_rig: str):
    options = _load_rig_config(args.rig_config) if args.rig_config else {}
    return load_rig(args.rig or dataset_rig, **options).mapping


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", type=Path, required=True, help="collected dataset directory")
    p.add_argument("--checkpoint", type=Path, help="V-JEPA 2-AC weights (default: the torch.hub cache)")
    p.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    p.add_argument("--batch-size", type=int, default=16)


def _add_rig_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rig", help="rig whose V-JEPA mapping to use (default: the rig the dataset was collected with)")
    p.add_argument("--rig-config", type=Path, help="JSON (or YAML) options for the rig's factory")


def _add_encode(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "encode",
        help="encode a dataset's photos once with the frozen V-JEPA encoder",
        description="Write <data>/latents/latents.pt; reruns only encode new frames.",
    )
    _add_model_args(p)
    p.set_defaults(handler=_encode)


def _encode(args: argparse.Namespace) -> int:
    from vjepa_byor.predictor import encode_dataset, loading

    read_meta(args.data)
    device = _device(args.device)
    encoder = loading.load_pretrained(args.checkpoint, predictor=False).encoder.to(device)

    def progress(done: int, total: int) -> None:
        print(f"\rencoded {done}/{total} new frames", end="", flush=True)

    cache = encode_dataset(args.data, encoder, _model_id(args.checkpoint), batch_size=args.batch_size, on_progress=progress)
    print(f"\nlatent cache: {len(cache.frames)} frames x {cache.tokens} tokens x {cache.dim}")
    return 0


def _add_finetune(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "finetune",
        help="fine-tune the V-JEPA 2-AC predictor on an encoded dataset",
        description="Train on the dataset's train split, keep the best epoch on val, save it to --out.",
    )
    _add_model_args(p)
    _add_rig_args(p)
    p.add_argument("--out", type=Path, required=True, help="directory for the fine-tuned predictor")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, help="peak learning rate (default 2e-5, or 2e-4 with --lora-rank)")
    p.add_argument("--lora-rank", type=int, default=0, help="train LoRA adapters of this rank instead of all weights")
    p.add_argument("--patience", type=int, help="stop after this many epochs without a better val loss")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-vram-gb", type=float, help="cap this process's GPU memory, e.g. 8 to check it fits an 8 GB card")
    p.set_defaults(handler=_finetune)


def _finetune(args: argparse.Namespace) -> int:
    from vjepa_byor.predictor import FinetuneConfig, LatentDataset, finetune, load_cache, loading

    meta = read_meta(args.data)
    mapping = _rig_mapping(args, meta.rig)
    device = _device(args.device)
    if args.max_vram_gb is not None and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb * 2**30 / total), index)

    cache = load_cache(args.data)
    train = LatentDataset(args.data, mapping, "train", cache=cache)
    val = LatentDataset(args.data, mapping, "val", cache=cache)
    print(f"train {len(train)} samples, val {len(val)} samples")
    predictor = loading.load_pretrained(args.checkpoint, encoder=False).predictor
    config = FinetuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        patience=args.patience,
        seed=args.seed,
    )

    def log(e) -> None:
        train_loss = "-" if e.train_loss is None else f"{e.train_loss:.4f}"
        val_loss = "-" if e.val_loss is None else f"{e.val_loss:.4f}"
        print(f"epoch {e.epoch:3d}  train {train_loss}  val {val_loss}  lr {e.lr:.2e}  {e.seconds:.0f}s", flush=True)

    info = {
        "dataset": str(args.data.resolve()),
        "rig": args.rig or meta.rig,
        "base_model": _model_id(args.checkpoint),
        "latents_model": cache.model,
    }
    result = finetune(predictor, train, val, config, args.out, device=device, info=info, on_epoch=log)
    print(f"best epoch {result.best_epoch} (val {result.best_val_loss:.4f}); saved to {result.directory}")
    if result.peak_vram_bytes is not None:
        print(f"peak VRAM {result.peak_vram_bytes / 2**30:.2f} GiB")
    return 0


def _add_eval(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "eval",
        help="compare a fine-tuned predictor with the original one and a no-change baseline",
        description="Mean L1 error between predicted and true next-photo latents on a held-out split.",
    )
    _add_model_args(p)
    _add_rig_args(p)
    p.add_argument("--finetuned", type=Path, required=True, help="directory written by vjepa-byor finetune")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--json", type=Path, help="also write the report here")
    p.set_defaults(handler=_eval)


def _eval(args: argparse.Namespace) -> int:
    from vjepa_byor.predictor import LatentDataset, evaluate, loading

    meta = read_meta(args.data)
    info = loading.read_finetuned_info(args.finetuned)
    args.rig = args.rig or info.get("rig")
    mapping = _rig_mapping(args, meta.rig)
    device = _device(args.device)
    dataset = LatentDataset(args.data, mapping, args.split)

    models = {}
    models["original"] = loading.load_pretrained(args.checkpoint, encoder=False).predictor.to(device)
    report_original = evaluate(dataset, models, batch_size=args.batch_size, device=device)
    del models["original"]
    models["fine-tuned"] = loading.load_finetuned(args.finetuned).to(device)
    report = evaluate(dataset, models, batch_size=args.batch_size, device=device)
    report.errors = {**report_original.errors, **report.errors}

    print(f"{args.split} split: " + report.format())
    passed = report.beats("fine-tuned", "no change") and report.beats("fine-tuned", "original")
    print(f"fine-tuned beats both on the samples where the rig moved: {'yes' if passed else 'NO'}")
    if args.json is not None:
        record = {"split": args.split, "samples": report.samples, "changed": report.changed, "errors": report.errors}
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vjepa-byor", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_collect(subparsers)
    _add_encode(subparsers)
    _add_finetune(subparsers)
    _add_eval(subparsers)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
