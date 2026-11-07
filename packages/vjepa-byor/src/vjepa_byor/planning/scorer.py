"""Scores candidate actions: how far each predicted next state is from the goal state.

All states are V-JEPA latents ``(tokens, dim)``. A scorer takes the predicted
latents of every candidate ``(N, tokens, dim)``, the goal photo's latents and
the current photo's latents, and returns one error per candidate ``(N,)``.
Lower is better, and the same inputs always give the same errors::

    predicted = model.predict(z_current, state.as_tensor(), actions)
    scores = score_actions(L1Scorer(), predicted, z_goal, z_current, labels)
    scores.best_label, scores.errors

:class:`L1Scorer` is the energy V-JEPA 2-AC plans with: the mean L1 distance
over every token. :class:`ChangeWeightedL1Scorer` puts more weight on the
tokens where the goal photo differs from the current one, so a small object
moving across a large static scene is not drowned out by the background.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Scorer(Protocol):
    """Maps predicted latents ``(N, tokens, dim)`` and a goal ``(tokens, dim)`` to errors ``(N,)``; lower is better."""

    name: str

    def __call__(self, predicted: torch.Tensor, goal: torch.Tensor, current: torch.Tensor) -> torch.Tensor: ...


def _check(predicted: torch.Tensor, goal: torch.Tensor, current: torch.Tensor) -> None:
    if predicted.ndim != 3 or goal.shape != predicted.shape[1:] or current.shape != goal.shape:
        raise ValueError(
            "expected predicted (N, tokens, dim) and goal and current (tokens, dim), got "
            f"{tuple(predicted.shape)}, {tuple(goal.shape)} and {tuple(current.shape)}"
        )


def token_errors(predicted: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Mean L1 error of each token, ``(N, tokens)``, in float32."""
    return (predicted.float() - goal.float()).abs().mean(dim=-1)


@dataclass(frozen=True)
class L1Scorer:
    """Mean L1 distance between each prediction and the goal over all tokens."""

    name: str = "l1"

    def __call__(self, predicted: torch.Tensor, goal: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        _check(predicted, goal, current)
        return token_errors(predicted, goal.to(predicted.device)).mean(dim=-1)


@dataclass(frozen=True)
class ChangeWeightedL1Scorer:
    """Token-weighted L1: tokens where the goal differs from the current photo count more.

    Each token's weight mixes a uniform share (``1 - alpha``) with its share of
    the change between the current and the goal latents (``alpha``). With
    ``alpha=0`` this is :class:`L1Scorer`. When the goal equals the current
    photo there is no change to weight by, and the weights are uniform.
    """

    alpha: float = 0.9
    name: str = "change-weighted"

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")

    def weights(self, goal: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Per-token weights ``(tokens,)`` that sum to 1."""
        change = (goal.float() - current.float()).abs().mean(dim=-1)
        uniform = torch.full_like(change, 1 / change.numel())
        total = change.sum()
        if total <= 0:
            return uniform
        return (1 - self.alpha) * uniform + self.alpha * change / total

    def __call__(self, predicted: torch.Tensor, goal: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        _check(predicted, goal, current)
        goal, current = goal.to(predicted.device), current.to(predicted.device)
        return token_errors(predicted, goal) @ self.weights(goal, current)


SCORERS = {"l1": L1Scorer, "change-weighted": ChangeWeightedL1Scorer}


def make_scorer(name: str, **options: float) -> Scorer:
    """A scorer by name (``l1`` or ``change-weighted``)."""
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; choose from {', '.join(SCORERS)}")
    return SCORERS[name](**options)


@dataclass(frozen=True)
class ActionScores:
    """One error per candidate action, in the action space's order."""

    errors: torch.Tensor  # float32 (N,), on the CPU
    labels: list[str]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.errors):
            raise ValueError(f"{len(self.labels)} labels for {len(self.errors)} errors")

    def order(self) -> torch.Tensor:
        """Candidate indices from best (lowest error) to worst; ties keep the action space's order."""
        return torch.argsort(self.errors, stable=True)

    @property
    def best(self) -> int:
        return int(self.order()[0])

    @property
    def best_label(self) -> str:
        return self.labels[self.best]

    @property
    def best_error(self) -> float:
        return float(self.errors[self.best])

    def rank(self, index: int) -> int:
        """1-based rank of candidate ``index`` (1 = best)."""
        return int((self.order() == index).nonzero()[0, 0]) + 1

    def ranked(self) -> list[tuple[str, float]]:
        return [(self.labels[i], float(self.errors[i])) for i in self.order().tolist()]


def score_actions(
    scorer: Scorer,
    predicted: torch.Tensor,
    goal: torch.Tensor,
    current: torch.Tensor,
    labels: Sequence[str],
) -> ActionScores:
    """Run ``scorer`` and pair its errors with the action labels."""
    return ActionScores(scorer(predicted, goal, current).float().cpu(), list(labels))
