"""Reading collected datasets: a torch Dataset, episode-level splits, export and verification."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vjepa_byor.data.sample import STATUSES, EpisodeInfo, Sample, frame_path
from vjepa_byor.data.storage import (
    EVENTS,
    MANIFEST,
    DatasetFormatError,
    atomic_write,
    frame_size,
    load_frame,
    manifest_is_truncated,
    read_meta,
    read_samples,
    write_manifest,
    write_meta,
)

SPLITS = "splits.json"
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


# Splits


def _hash_split(episode_id: str, seed: int, ratios: Sequence[float]) -> str:
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name, ratio in zip(SPLIT_NAMES, ratios):
        cumulative += ratio
        if u < cumulative:
            return name
    return SPLIT_NAMES[-1]


def _initial_split(episodes: list[str], seed: int, ratios: Sequence[float]) -> dict[str, str]:
    order = sorted(episodes)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_val = round(n * ratios[1])
    n_test = round(n * ratios[2])
    if n >= 3:  # every split gets at least one episode when there are enough
        n_val, n_test = max(n_val, 1 if ratios[1] else 0), max(n_test, 1 if ratios[2] else 0)
    n_train = n - n_val - n_test
    names = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return dict(zip(order, names))


def ensure_splits(
    root: str | Path, *, seed: int = 0, ratios: Sequence[float] = DEFAULT_RATIOS
) -> dict[str, str]:
    """Episode -> split, stored in ``splits.json`` so it never changes.

    The first call shuffles the episodes with ``seed`` and cuts them in
    ``ratios``. Later calls keep every stored assignment; episodes added since
    (by a resumed run) are placed by hashing their id with the stored seed. The
    stored seed and ratios win over the arguments.
    """
    root = Path(root)
    if abs(sum(ratios) - 1) > 1e-9 or len(ratios) != 3:
        raise ValueError("ratios must be three numbers (train, val, test) summing to 1")
    episodes = sorted({s.episode_id for s in read_samples(root)})
    path = root / SPLITS
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        seed, ratios, assigned = stored["seed"], stored["ratios"], stored["episodes"]
        new = [e for e in episodes if e not in assigned]
        if not new:
            return dict(assigned)
        assigned.update({e: _hash_split(e, seed, ratios) for e in new})
    else:
        assigned = _initial_split(episodes, seed, ratios)
    record = {"seed": seed, "ratios": list(ratios), "episodes": dict(sorted(assigned.items()))}
    atomic_write(path, (json.dumps(record, indent=2) + "\n").encode())
    return dict(assigned)


# Filtering


def select_samples(
    root: str | Path,
    *,
    split: str | None = None,
    include_statuses: Iterable[str] | None = None,
    drop_no_change: bool = False,
    seed: int = 0,
) -> list[Sample]:
    samples = read_samples(root)
    if split is not None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, not {split!r}")
        splits = ensure_splits(root, seed=seed)
        samples = [s for s in samples if splits.get(s.episode_id) == split]
    if include_statuses is not None:
        allowed = set(include_statuses)
        if unknown := allowed - set(STATUSES):
            raise ValueError(f"unknown statuses {sorted(unknown)}; known: {STATUSES}")
        samples = [s for s in samples if s.status in allowed]
    if drop_no_change:
        samples = [s for s in samples if s.changed]
    return samples


def episodes(root: str | Path) -> list[EpisodeInfo]:
    counts: dict[str, Counter[str]] = {}
    for s in read_samples(root):
        c = counts.setdefault(s.episode_id, Counter())
        c["samples"] += 1
        c["changed"] += s.changed
        c["rejected"] += s.status != "ok"
    return [EpisodeInfo(e, c["samples"], c["changed"], c["rejected"]) for e, c in sorted(counts.items())]


# Torch dataset


class CollectedDataset(Dataset):
    """Samples of a collected dataset, images loaded lazily.

    Each item is a dict with ``before`` and ``after`` (``uint8`` tensors of
    shape ``(3, H, W)``, exactly as stored), ``state_before``, ``state_after``
    and ``action`` (float tensors) and ``sample`` (the :class:`Sample`).
    Nothing is resized here; pass ``transform`` (called on the item dict) for
    that.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        *,
        include_statuses: Iterable[str] | None = None,
        drop_no_change: bool = False,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.meta = read_meta(self.root)
        self.samples = select_samples(
            self.root,
            split=split,
            include_statuses=include_statuses,
            drop_no_change=drop_no_change,
            seed=seed,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        s = self.samples[index]
        item = {
            "before": load_frame(self.root / s.before_frame),
            "after": load_frame(self.root / s.after_frame),
            "state_before": torch.tensor(s.state_before_tensor, dtype=torch.float32),
            "state_after": torch.tensor(s.state_after_tensor, dtype=torch.float32),
            "action": torch.tensor(s.action, dtype=torch.float32),
            "sample": s,
        }
        return self.transform(item) if self.transform is not None else item


# Export


def export(
    root: str | Path,
    out: str | Path,
    *,
    split: str | None = None,
    include_statuses: Iterable[str] | None = None,
    drop_no_change: bool = False,
    seed: int = 0,
) -> int:
    """Copy the chosen samples and their frames into a new, self-contained dataset.

    The copy has the same layout and passes :func:`verify`. Returns the number
    of samples exported.
    """
    root, out = Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    meta = read_meta(root)
    samples = select_samples(
        root, split=split, include_statuses=include_statuses, drop_no_change=drop_no_change, seed=seed
    )
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted({f for s in samples for f in (s.before_frame, s.after_frame)})
    for rel in frames:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, out / rel)
    extra = dict(meta.extra, exported_from=str(root.resolve()), split=split)
    exported_meta = type(meta).from_dict({**meta.to_dict(), "extra": extra})
    write_meta(out, exported_meta)
    write_manifest(out, samples)
    (out / EVENTS).touch()
    return len(samples)


# Verification


@dataclass
class VerifyReport:
    samples: int = 0
    episodes: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = [f"{'OK' if self.ok else 'FAILED'}: {self.samples} samples in {self.episodes} episodes"]
        lines += [f"  error: {e}" for e in self.errors[:50]]
        if len(self.errors) > 50:
            lines.append(f"  ... and {len(self.errors) - 50} more errors")
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def verify(root: str | Path) -> VerifyReport:
    """Check a dataset on disk is complete and consistent.

    Every manifest line's frames exist and match the resolution in
    ``meta.json``; sample ids are unique; frame paths follow the layout; and
    consecutive steps of an episode chain (step k's after frame and state are
    step k+1's before frame and state).
    """
    root = Path(root)
    report = VerifyReport()
    try:
        meta = read_meta(root)
    except DatasetFormatError as e:
        report.errors.append(str(e))
        return report
    try:
        samples = read_samples(root)
    except DatasetFormatError as e:
        report.errors.append(str(e))
        return report
    if manifest_is_truncated(root):
        report.warnings.append(f"{MANIFEST} ends in a truncated line (dropped on the next resume)")

    report.samples = len(samples)
    expected = tuple(meta.resolution)
    checked: dict[str, tuple[int, int] | None] = {}
    by_step: dict[tuple[str, int], Sample] = {}
    for s in samples:
        for rel, want in (
            (s.before_frame, frame_path(s.episode_id, s.step_index)),
            (s.after_frame, frame_path(s.episode_id, s.step_index + 1)),
        ):
            if rel != want:
                report.errors.append(f"{s.sample_id}: frame {rel} should be {want}")
            if rel not in checked:
                path = root / rel
                checked[rel] = frame_size(path) if path.is_file() else None
            size = checked[rel]
            if size is None:
                report.errors.append(f"{s.sample_id}: frame {rel} is missing")
            elif size != expected:
                report.errors.append(
                    f"{s.sample_id}: frame {rel} is {size[0]}x{size[1]} (HxW), meta.json says {expected[0]}x{expected[1]}"
                )
        if list(s.resolution) != list(meta.resolution):
            report.errors.append(f"{s.sample_id}: resolution {s.resolution} differs from meta.json {meta.resolution}")
        if s.status not in STATUSES:
            report.errors.append(f"{s.sample_id}: unknown status {s.status!r}")
        if (s.episode_id, s.step_index) in by_step:
            report.errors.append(f"{s.sample_id}: step {s.step_index} of {s.episode_id} appears twice")
        by_step[(s.episode_id, s.step_index)] = s

    for sample_id, n in Counter(s.sample_id for s in samples).items():
        if n > 1:
            report.errors.append(f"sample id {sample_id} appears {n} times")

    for (episode, step), s in by_step.items():
        nxt = by_step.get((episode, step + 1))
        if nxt is None:
            continue
        if s.after_frame != nxt.before_frame:
            report.errors.append(
                f"{episode}: step {step} ends on {s.after_frame} but step {step + 1} starts on {nxt.before_frame}"
            )
        if s.state_after != nxt.state_before:
            report.errors.append(f"{episode}: state after step {step} is not the state before step {step + 1}")

    report.episodes = len({s.episode_id for s in samples})
    return report
