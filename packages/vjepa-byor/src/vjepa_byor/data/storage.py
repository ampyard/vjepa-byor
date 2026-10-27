"""On-disk dataset format. This is a contract: ``encode`` and ``finetune`` read it.

Bump :data:`~vjepa_byor.data.sample.SCHEMA_VERSION` on any change to it.

Layout::

    <root>/
    ├── meta.json        # DatasetMeta: schema_version, rig, state_type, action vocabulary,
    │                    #   resolution [height, width], rig config hash, collector version, created_at
    ├── manifest.jsonl   # one Sample per line, append-only
    ├── events.jsonl     # rejections, failures, syncs, resumptions; append-only, for humans
    ├── splits.json      # episode -> train/val/test, written by the dataset reader
    └── frames/
        └── ep000012/
            ├── 000.png  # frame 0 = before of step 0
            ├── 001.png  # frame 1 = after of step 0 = before of step 1
            └── ...

Frames are lossless RGB PNGs exactly as captured; nothing is resized here.
Step ``k`` of an episode uses frame ``k`` as its before photo and frame ``k+1``
as its after photo, so ``n`` steps store ``n+1`` frames. Manifest paths are
relative to the root and use forward slashes.

Write order makes interrupted runs safe to resume: a frame is written to a
temporary file and renamed into place, and only then is the manifest line that
references it appended, flushed and fsynced. A manifest line never points at a
missing frame. A crash can leave a truncated last line (dropped on resume) and
frames no line references (ignored).

Readers refuse a ``meta.json`` whose ``schema_version`` they do not know.
"""

from __future__ import annotations

import ctypes
import io
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vjepa_byor.data.sample import SCHEMA_VERSION, DatasetMeta, Sample, frame_path

META = "meta.json"
MANIFEST = "manifest.jsonl"
EVENTS = "events.jsonl"
FRAMES = "frames"

_EPISODE_DIR = re.compile(r"^ep(\d+)$")


class DatasetFormatError(ValueError):
    """A dataset on disk is missing, corrupt, or in an unknown format."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# Images


def encode_png(image: torch.Tensor) -> bytes:
    """``uint8`` tensor of shape ``(3, H, W)`` -> lossless PNG bytes."""
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a uint8 (3, H, W) image, got {image.dtype} {tuple(image.shape)}")
    _, height, width = image.shape
    hwc = image.detach().cpu().permute(1, 2, 0).contiguous()
    pixels = ctypes.string_at(hwc.data_ptr(), hwc.numel())  # torch has no buffer protocol; avoids needing numpy
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def load_frame(path: str | Path) -> torch.Tensor:
    """Read a PNG as a ``uint8`` tensor of shape ``(3, H, W)``."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        width, height = rgb.size
        data = bytearray(rgb.tobytes())
    return torch.frombuffer(data, dtype=torch.uint8).view(height, width, 3).permute(2, 0, 1).contiguous()


def frame_size(path: str | Path) -> tuple[int, int]:
    """``(height, width)`` of a PNG, reading only its header."""
    with Image.open(path) as im:
        width, height = im.size
    return height, width


# Reading


def read_meta(root: str | Path) -> DatasetMeta:
    path = Path(root) / META
    if not path.is_file():
        raise DatasetFormatError(f"{root} is not a dataset: {META} is missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DatasetFormatError(f"{path} is not valid JSON: {e}") from e
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DatasetFormatError(
            f"{path} has schema_version {version!r}; this version of vjepa-byor reads {SCHEMA_VERSION}"
        )
    return DatasetMeta.from_dict(data)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int | None]:
    """Parsed lines, plus the byte offset of a truncated last line (``None`` if intact)."""
    if not path.is_file():
        return [], None
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        complete = line.endswith(b"\n")
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            if not complete:
                return records, offset
            raise DatasetFormatError(f"{path}: corrupt line at byte {offset}: {e}") from e
        if not complete:
            # Valid JSON but no newline: the write was cut off just before it.
            records.pop()
            return records, offset
        offset += len(line)
    return records, None


def read_samples(root: str | Path) -> list[Sample]:
    """Every complete manifest line. A truncated last line is skipped."""
    records, _ = _read_jsonl(Path(root) / MANIFEST)
    return [Sample.from_dict(r) for r in records]


def manifest_is_truncated(root: str | Path) -> bool:
    """Whether the manifest ends in a line cut off by a crash."""
    return _read_jsonl(Path(root) / MANIFEST)[1] is not None


def write_meta(root: str | Path, meta: DatasetMeta) -> None:
    atomic_write(Path(root) / META, (json.dumps(meta.to_dict(), indent=2) + "\n").encode())


def write_manifest(root: str | Path, samples: list[Sample]) -> None:
    """Write a whole manifest at once (for exports; collection appends with :class:`DatasetWriter`)."""
    lines = "".join(json.dumps(s.to_dict(), separators=(",", ":")) + "\n" for s in samples)
    atomic_write(Path(root) / MANIFEST, lines.encode())


def read_events(root: str | Path) -> list[dict[str, Any]]:
    records, _ = _read_jsonl(Path(root) / EVENTS)
    return records


def next_episode_number(root: str | Path) -> int:
    """The first episode number not used by the manifest or by any frame directory."""
    used = [-1]
    frames = Path(root) / FRAMES
    if frames.is_dir():
        used += [int(m[1]) for p in frames.iterdir() if (m := _EPISODE_DIR.match(p.name))]
    used += [int(m[1]) for s in read_samples(root) if (m := _EPISODE_DIR.match(s.episode_id))]
    return max(used) + 1


# Writing


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return  # not supported on Windows
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class DatasetWriter:
    """Appends frames, samples and events to a dataset. The only code that writes the layout.

    Use :meth:`create` for a new dataset and :meth:`resume` to add to an
    existing one.
    """

    def __init__(self, root: Path, meta: DatasetMeta) -> None:
        self.root = root
        self.meta = meta
        self._manifest = open(root / MANIFEST, "ab")
        self._events = open(root / EVENTS, "ab")

    @classmethod
    def create(cls, root: str | Path, meta: DatasetMeta) -> DatasetWriter:
        root = Path(root)
        if (root / META).exists() or (root / MANIFEST).exists():
            raise FileExistsError(f"{root} already holds a dataset; resume it instead")
        root.mkdir(parents=True, exist_ok=True)
        write_meta(root, meta)
        writer = cls(root, meta)
        writer.log_event("created", rig=meta.rig)
        return writer

    @classmethod
    def resume(cls, root: str | Path, meta: DatasetMeta) -> DatasetWriter:
        """Open an existing dataset for appending, after checking ``meta`` matches it.

        Raises :class:`DatasetFormatError` if the dataset was collected with a
        different rig, vocabulary or resolution. A truncated last manifest line
        is cut off and logged.
        """
        root = Path(root)
        existing = read_meta(root)
        if problems := existing.incompatibilities(meta):
            raise DatasetFormatError(f"cannot resume {root}: " + "; ".join(problems))
        dropped = cls._drop_truncated_tail(root / MANIFEST)
        cls._drop_truncated_tail(root / EVENTS)
        writer = cls(root, existing)
        if dropped is not None:
            writer.log_event("truncated_line_dropped", file=MANIFEST, content=dropped)
        notes = {}
        if existing.rig_config_hash != meta.rig_config_hash:
            notes["warning"] = "rig config differs from the one the dataset was started with"
        writer.log_event("resumed", samples=len(read_samples(root)), **notes)
        return writer

    @staticmethod
    def _drop_truncated_tail(path: Path) -> str | None:
        _, offset = _read_jsonl(path)
        if offset is None:
            return None
        with open(path, "r+b") as f:
            f.seek(offset)
            tail = f.read().decode("utf-8", errors="replace")
            f.truncate(offset)
        return tail

    def write_frame(self, episode_id: str, index: int, image: torch.Tensor) -> str:
        """Store one frame atomically; returns its path relative to the root."""
        height, width = self.meta.resolution
        if tuple(image.shape) != (3, height, width):
            raise ValueError(f"frame shape {tuple(image.shape)} does not match the dataset's (3, {height}, {width})")
        rel = frame_path(episode_id, index)
        atomic_write(self.root / rel, encode_png(image))
        return rel

    def append_sample(self, sample: Sample) -> None:
        for rel in (sample.before_frame, sample.after_frame):
            if not (self.root / rel).is_file():
                raise FileNotFoundError(f"refusing to record {sample.sample_id}: {rel} is not on disk")
        self._append(self._manifest, sample.to_dict())

    def log_event(self, kind: str, **fields: Any) -> None:
        self._append(self._events, {"time": now(), "event": kind, **fields})

    @staticmethod
    def _append(f: Any, record: dict[str, Any]) -> None:
        f.write((json.dumps(record, separators=(",", ":")) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())

    def close(self) -> None:
        self._manifest.close()
        self._events.close()

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
