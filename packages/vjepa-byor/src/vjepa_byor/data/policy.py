"""How the collector picks the next action."""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

import torch

from vjepa_byor.interfaces import RigState


@runtime_checkable
class ActionPolicy(Protocol):
    """Picks actions by index into the rig's vocabulary.

    The collector owns the random generator (seeded per episode, so resumed
    runs stay reproducible) and passes it in.
    """

    def choose(self, state: RigState, rng: random.Random) -> int: ...

    def observe(self, state: RigState, action_index: int, result: RigState | None) -> None:
        """Called after every move; ``result`` is ``None`` if the rig rejected it."""
        ...


class EdgeAwareRandomWalk:
    """A random walk that stays inside the rig's limits and drifts back from its edges.

    The core library knows neither a rig's limits nor what its actions do to
    its state, so this policy works from two sources:

    - ``bounds``: optional ``(low, high)`` limits on ``RigState.as_tensor()``,
      inclusive. Without them, the walk only avoids moves it has already seen
      rejected from the same state.
    - What it observes: the mean change in the state tensor each action has
      produced so far. This predicts where an action leads, for rigs whose
      actions are relative moves; an action not yet tried is assumed safe.

    Each step it samples uniformly from actions predicted to stay inside the
    bounds. Within ``margin`` (a fraction of the range) of an edge it picks,
    with probability ``edge_bias``, from those that move toward the centre.
    With probability ``reject_rate`` it deliberately picks an action predicted
    to be rejected, so the dataset shows the model what happens at a limit.
    """

    def __init__(
        self,
        actions: torch.Tensor,
        *,
        bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        margin: float = 0.1,
        edge_bias: float = 0.75,
        reject_rate: float = 0.02,
    ) -> None:
        self.num_actions = len(actions)
        if bounds is not None:
            low, high = (torch.as_tensor(b, dtype=torch.float64).flatten() for b in bounds)
            if low.shape != high.shape or bool((low > high).any()):
                raise ValueError("bounds must be (low, high) with low <= high in every dimension")
            bounds = (low, high)
        self.bounds = bounds
        self.margin = margin
        self.edge_bias = edge_bias
        self.reject_rate = reject_rate
        self._delta_sum: dict[int, torch.Tensor] = {}
        self._delta_count: dict[int, int] = {}
        self._rejected: set[tuple[tuple[float, ...], int]] = set()

    @staticmethod
    def _key(state: torch.Tensor) -> tuple[float, ...]:
        return tuple(state.to(torch.float64).tolist())

    def predict(self, state: RigState, action_index: int) -> torch.Tensor | None:
        """Expected state tensor after the action, or ``None`` if it has not been tried."""
        if action_index not in self._delta_count:
            return None
        delta = self._delta_sum[action_index] / self._delta_count[action_index]
        return state.as_tensor().to(torch.float64) + delta

    def _in_bounds(self, point: torch.Tensor) -> bool:
        assert self.bounds is not None
        low, high = self.bounds
        return bool(((point >= low) & (point <= high)).all())

    def would_be_rejected(self, state: RigState, action_index: int) -> bool:
        if (self._key(state.as_tensor()), action_index) in self._rejected:
            return True
        if self.bounds is None:
            return False
        predicted = self.predict(state, action_index)
        return predicted is not None and not self._in_bounds(predicted)

    def _near_edge(self, point: torch.Tensor) -> bool:
        assert self.bounds is not None
        low, high = self.bounds
        slack = (high - low) * self.margin
        return bool(((point - low < slack) | (high - point < slack)).any())

    def _toward_centre(self, state: RigState, candidates: list[int]) -> list[int]:
        assert self.bounds is not None
        low, high = self.bounds
        here = state.as_tensor().to(torch.float64)
        centre = (low + high) / 2
        scale = torch.where(high > low, high - low, torch.ones_like(low))
        distance = float(((here - centre) / scale).norm())
        toward = []
        for a in candidates:
            predicted = self.predict(state, a)
            if predicted is not None and float(((predicted - centre) / scale).norm()) < distance:
                toward.append(a)
        return toward

    def choose(self, state: RigState, rng: random.Random) -> int:
        everything = range(self.num_actions)
        rejected = [a for a in everything if self.would_be_rejected(state, a)]
        allowed = [a for a in everything if a not in rejected]
        if rejected and rng.random() < self.reject_rate:
            return rng.choice(rejected)
        if not allowed:
            return rng.choice(list(everything))
        if self.bounds is not None and self._near_edge(state.as_tensor().to(torch.float64)):
            toward = self._toward_centre(state, allowed)
            if toward and rng.random() < self.edge_bias:
                return rng.choice(toward)
        return rng.choice(allowed)

    def observe(self, state: RigState, action_index: int, result: RigState | None) -> None:
        before = state.as_tensor().to(torch.float64)
        if result is None:
            self._rejected.add((self._key(before), action_index))
            return
        delta = result.as_tensor().to(torch.float64) - before
        if action_index in self._delta_sum:
            self._delta_sum[action_index] += delta
            self._delta_count[action_index] += 1
        else:
            self._delta_sum[action_index] = delta.clone()
            self._delta_count[action_index] = 1
