"""Checks a scorer on labelled data: does the action the rig really took rank first?

For each held-out sample where the rig moved, the predictor imagines every
action in the basket from the before photo and rig state, the scorer compares
each prediction with the after photo (the goal), and we record where the true
action ranks. A good scorer puts it first, or at least near the top.

Actions with the same rank-1 direction but a different step size are counted
separately (``direction``), because telling 1 mm from 5 mm is harder than
telling north from south.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from vjepa_byor.planning.scorer import Scorer, score_actions
from vjepa_byor.predictor.inference import predict_actions
from vjepa_byor.predictor.latent_cache import LatentDataset


@dataclass
class RankingReport:
    scorer: str
    samples: int = 0
    top1: int = 0
    top3: int = 0
    same_direction: int = 0
    rank_sum: int = 0
    by_action: dict[str, list[int]] = field(default_factory=dict)
    """True action label -> its rank in every sample where it was taken."""

    def add(self, label: str, rank: int, same_direction: bool) -> None:
        self.samples += 1
        self.top1 += rank == 1
        self.top3 += rank <= 3
        self.same_direction += same_direction
        self.rank_sum += rank
        self.by_action.setdefault(label, []).append(rank)

    def rate(self, count: int) -> float:
        return count / self.samples if self.samples else float("nan")

    @property
    def mean_rank(self) -> float:
        return self.rank_sum / self.samples if self.samples else float("nan")

    def summary(self) -> dict[str, float | int | str]:
        return {
            "scorer": self.scorer,
            "samples": self.samples,
            "top1": self.rate(self.top1),
            "top3": self.rate(self.top3),
            "same_direction": self.rate(self.same_direction),
            "mean_rank": self.mean_rank,
        }

    def format(self, candidates: int | None = None) -> str:
        out_of = f" of {candidates}" if candidates else ""
        return (
            f"{self.scorer}: {self.samples} samples, true action ranked 1st {self.rate(self.top1):.0%}, "
            f"top 3 {self.rate(self.top3):.0%}, best has the true direction {self.rate(self.same_direction):.0%}, "
            f"mean rank {self.mean_rank:.1f}{out_of}"
        )


def direction_of(label: str) -> str:
    """``"NE 5mm"`` -> ``"NE"``; labels without a space are their own direction."""
    return label.split(" ", 1)[0]


@torch.no_grad()
def rank_true_actions(
    predictor: nn.Module,
    dataset: LatentDataset,
    actions: torch.Tensor,
    labels: Sequence[str],
    scorers: Sequence[Scorer],
    *,
    batch_size: int = 64,
) -> list[RankingReport]:
    """Rank the true action among ``actions`` for each sample of ``dataset`` where the rig moved.

    ``predictor`` should already be on its device; ``dataset`` must use the rig's
    mapping. One report per scorer, all from the same predictions.
    """
    predictor.eval()
    reports = [RankingReport(s.name) for s in scorers]
    for i, sample in enumerate(dataset.samples):
        if not sample.changed:
            continue
        item = dataset[i]
        predicted = predict_actions(
            predictor,
            dataset.mapping,
            item["z_before"],
            torch.tensor(sample.state_before_tensor),
            actions,
            batch_size=batch_size,
        )
        for scorer, report in zip(scorers, reports, strict=True):
            scores = score_actions(scorer, predicted, item["z_after"], item["z_before"], labels)
            rank = scores.rank(sample.action_index)
            report.add(sample.action_name, rank, direction_of(scores.best_label) == direction_of(sample.action_name))
    return reports
