"""Choosing actions with a fine-tuned predictor: score every candidate against a goal photo.

:mod:`~vjepa_byor.planning.scorer` turns predicted latents into one error per
action; :mod:`~vjepa_byor.planning.ranking` checks a scorer on labelled data.
"""

from vjepa_byor.planning.ranking import RankingReport, direction_of, rank_true_actions
from vjepa_byor.planning.scorer import (
    SCORERS,
    ActionScores,
    ChangeWeightedL1Scorer,
    L1Scorer,
    Scorer,
    make_scorer,
    score_actions,
    token_errors,
)

__all__ = [
    "SCORERS",
    "ActionScores",
    "ChangeWeightedL1Scorer",
    "L1Scorer",
    "RankingReport",
    "Scorer",
    "direction_of",
    "make_scorer",
    "rank_true_actions",
    "score_actions",
    "token_errors",
]
