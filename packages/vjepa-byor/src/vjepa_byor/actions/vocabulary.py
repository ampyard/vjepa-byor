"""Discrete action sets as tensors: the basket the planner scores."""

from __future__ import annotations

from collections.abc import Sequence

import torch

# Unit vectors on a plane, in compass order starting north (+y) and turning clockwise
COMPASS = {
    "N": (0, 1),
    "NE": (1, 1),
    "E": (1, 0),
    "SE": (1, -1),
    "S": (0, -1),
    "SW": (-1, -1),
    "W": (-1, 0),
    "NW": (-1, 1),
}


class DiscreteActionSpace:
    """A fixed list of actions with unique labels; implements :class:`~vjepa_byor.interfaces.ActionSpace`.

    Membership is exact: an action belongs only if it equals a row bit for bit,
    so choose rows a codec can reproduce exactly (whole numbers are safest).
    """

    def __init__(self, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> None:
        self._actions = torch.tensor([list(row) for row in rows], dtype=torch.float32)
        self._labels = list(labels)
        if self._actions.ndim != 2 or len(self._actions) == 0:
            raise ValueError("an action space needs at least one action, all of the same length")
        if len(self._labels) != len(self._actions):
            raise ValueError(f"{len(self._actions)} actions but {len(self._labels)} labels")
        if len(set(self._labels)) != len(self._labels):
            raise ValueError("action labels must be unique")
        if len(torch.unique(self._actions, dim=0)) != len(self._actions):
            raise ValueError("actions must be unique")
        if not torch.isfinite(self._actions).all():
            raise ValueError("actions must be finite")

    @classmethod
    def compass(cls, step_sizes: Sequence[int], unit: str = "") -> DiscreteActionSpace:
        """8 compass directions x ``step_sizes`` on a plane, as ``[dx, dy]`` rows.

        Diagonals move ``size`` along both axes. Rows are grouped by step size,
        smallest first, labelled e.g. ``"NE 5mm"`` for ``unit="mm"``.
        """
        if not step_sizes or any(s <= 0 for s in step_sizes):
            raise ValueError("step sizes must be positive")
        rows = [(dx * s, dy * s) for s in step_sizes for dx, dy in COMPASS.values()]
        labels = [f"{name} {s}{unit}" for s in step_sizes for name in COMPASS]
        return cls(rows, labels)

    @property
    def dim(self) -> int:
        return self._actions.shape[1]

    def actions(self) -> torch.Tensor:
        return self._actions.clone()

    def labels(self) -> list[str]:
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._actions)

    def index_of(self, action: torch.Tensor) -> int | None:
        """Row of ``action`` in :meth:`actions`, or ``None`` if it is not one of them."""
        if action.shape != (self.dim,):
            return None
        hits = (self._actions == action.to(self._actions.dtype)).all(dim=1).nonzero()
        return int(hits[0]) if len(hits) else None

    def contains(self, action: torch.Tensor) -> bool:
        return self.index_of(action) is not None

    def by_label(self, label: str) -> torch.Tensor:
        """The action labelled ``label``; raises ``KeyError`` if there is none."""
        try:
            return self._actions[self._labels.index(label)].clone()
        except ValueError:
            raise KeyError(label) from None
