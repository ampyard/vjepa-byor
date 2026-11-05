"""Record types for collected data: one transition, one dataset's metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Self

SCHEMA_VERSION = 1

STATUS_OK = "ok"
STATUS_REJECTED = "rejected"
STATUSES = (STATUS_OK, STATUS_REJECTED)


def episode_name(number: int) -> str:
    return f"ep{number:06d}"


def frame_path(episode_id: str, index: int) -> str:
    """Path of an episode's ``index``-th frame, relative to the dataset root."""
    return f"frames/{episode_id}/{index:03d}.png"


class _Record:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass(frozen=True)
class Sample(_Record):
    """One transition: before photo, rig state, action, after photo.

    ``status`` is ``ok`` when the rig executed the action and ``rejected`` when
    it reported a failure (for example an out-of-range move); a rejected action
    leaves the state unchanged. ``changed`` is ``state_after != state_before``,
    decided from the states the rig reported, not from the images.
    """

    sample_id: str
    episode_id: str
    step_index: int
    before_frame: str
    after_frame: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    state_before_tensor: list[float]
    state_after_tensor: list[float]
    action_index: int
    action_name: str
    action: list[float]
    command: str | None  # None for rigs driven without command lines (ActionTransport)
    status: str
    changed: bool
    resolution: list[int]
    timestamp: str
    error: str | None = None


@dataclass(frozen=True)
class DatasetMeta(_Record):
    """What a dataset was collected with; stored in ``meta.json``."""

    rig: str
    state_type: str
    action_labels: list[str]
    actions: list[list[float]]
    resolution: list[int]
    rig_config_hash: str
    collector_version: str
    created_at: str
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def incompatibilities(self, other: DatasetMeta) -> list[str]:
        """Reasons a run described by ``other`` cannot add to this dataset."""
        problems = []
        if self.rig != other.rig:
            problems.append(f"rig is {other.rig!r}, dataset was collected with {self.rig!r}")
        if self.action_labels != other.action_labels or self.actions != other.actions:
            problems.append("the rig's action vocabulary differs from the dataset's")
        if self.resolution != other.resolution:
            problems.append(f"camera resolution is {other.resolution}, dataset uses {self.resolution}")
        return problems


@dataclass(frozen=True)
class EpisodeInfo:
    """Summary of one episode, derived from the manifest."""

    episode_id: str
    num_samples: int
    num_changed: int
    num_rejected: int
