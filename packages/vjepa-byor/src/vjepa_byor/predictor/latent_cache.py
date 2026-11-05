"""Encodes a collected dataset's photos once with the frozen encoder, so fine-tuning never loads it.

The cache lives inside the dataset::

    <dataset>/latents/latents.pt   # {"model": id, "frames": [path, ...], "latents": bf16 (F, tokens, dim)}

Row ``i`` of ``latents`` is the layer-normalised encoding of ``frames[i]``
(paths as in the manifest). Each frame is encoded once even though it is the
after photo of one step and the before photo of the next. Running
:func:`encode_dataset` again after a resumed collection encodes only the new
frames; a different ``model`` id starts the cache afresh. The file is replaced
atomically, so an interrupted run leaves the previous cache intact.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from vjepa_byor.data.dataset import select_samples
from vjepa_byor.data.storage import load_frame, read_samples
from vjepa_byor.interfaces import VJepaMapping
from vjepa_byor.predictor.encoding import encode_frames

LATENTS_DIR = "latents"
LATENTS_FILE = "latents.pt"


@dataclass
class LatentCache:
    model: str
    frames: list[str]
    latents: torch.Tensor  # bf16 (num_frames, tokens, dim)

    def __post_init__(self) -> None:
        self.row = {path: i for i, path in enumerate(self.frames)}

    @property
    def tokens(self) -> int:
        return self.latents.shape[1]

    @property
    def dim(self) -> int:
        return self.latents.shape[2]

    def get(self, frame: str) -> torch.Tensor:
        return self.latents[self.row[frame]]


def cache_path(root: str | Path) -> Path:
    return Path(root) / LATENTS_DIR / LATENTS_FILE


def load_cache(root: str | Path) -> LatentCache:
    """The cache written by :func:`encode_dataset`, memory-mapped."""
    path = cache_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"{root} has no latent cache; run: vjepa-byor encode --data {root}")
    data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    return LatentCache(data["model"], list(data["frames"]), data["latents"])


def encode_dataset(
    root: str | Path,
    encoder: nn.Module,
    model: str,
    *,
    batch_size: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
) -> LatentCache:
    """Encode every frame the manifest references that the cache does not have yet.

    ``model`` identifies the encoder weights; a cache made with another id is
    discarded. ``on_progress(done, total)`` is called after each batch of new frames.
    """
    root = Path(root)
    wanted = sorted({f for s in read_samples(root) for f in (s.before_frame, s.after_frame)})
    old = None
    if cache_path(root).is_file():
        old = load_cache(root)
        if old.model != model:
            old = None
    have = set(old.frames) if old is not None else set()
    missing = [f for f in wanted if f not in have]
    if not missing and old is not None:
        return old

    chunks, done = [], 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        images = torch.stack([load_frame(root / f) for f in batch])
        chunks.append(encode_frames(encoder, images, batch_size=batch_size).to(torch.bfloat16).cpu())
        done += len(batch)
        if on_progress is not None:
            on_progress(done, len(missing))

    frames = (old.frames if old is not None else []) + missing
    parts = ([old.latents.clone()] if old is not None else []) + chunks
    latents = torch.cat(parts) if parts else torch.empty(0)
    path = cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model, "frames": frames, "latents": latents}, tmp)
    tmp.replace(path)
    return load_cache(root)


class LatentDataset(Dataset):
    """Samples of a collected dataset as predictor inputs and targets, from the latent cache.

    Each item is a dict with ``z_before`` and ``z_after`` (``float32 (tokens, dim)``)
    and ``action`` and ``state`` (``float32 (1, dim)``: one step, already passed
    through ``mapping``). Rejected moves are kept by default: their after photo
    shows no change, which is what the predictor should learn for them.
    """

    def __init__(
        self,
        root: str | Path,
        mapping: VJepaMapping,
        split: str | None = None,
        *,
        include_statuses: Iterable[str] | None = None,
        drop_no_change: bool = False,
        seed: int = 0,
        cache: LatentCache | None = None,
    ) -> None:
        self.root = Path(root)
        self.cache = cache if cache is not None else load_cache(self.root)
        self.samples = select_samples(
            self.root, split=split, include_statuses=include_statuses, drop_no_change=drop_no_change, seed=seed
        )
        if not self.samples:
            raise ValueError(f"no samples in {self.root} match (split={split!r})")
        missing = [s.sample_id for s in self.samples if s.before_frame not in self.cache.row or s.after_frame not in self.cache.row]
        if missing:
            raise ValueError(
                f"{len(missing)} samples have frames missing from the latent cache (first: {missing[0]}); "
                f"run: vjepa-byor encode --data {self.root}"
            )
        self.actions = mapping.map_actions(torch.tensor([s.action for s in self.samples], dtype=torch.float32))
        self.states = mapping.map_states(torch.tensor([s.state_before_tensor for s in self.samples], dtype=torch.float32))
        self.changed = torch.tensor([s.changed for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = self.samples[index]
        return {
            "z_before": self.cache.get(s.before_frame).float(),
            "z_after": self.cache.get(s.after_frame).float(),
            "action": self.actions[index].unsqueeze(0),
            "state": self.states[index].unsqueeze(0),
            "changed": self.changed[index],
        }
