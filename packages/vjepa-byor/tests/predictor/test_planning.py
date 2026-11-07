import pytest
import torch

from vjepa_byor.actions import DiscreteActionSpace
from vjepa_byor.planning import (
    ActionScores,
    ChangeWeightedL1Scorer,
    L1Scorer,
    Scorer,
    direction_of,
    make_scorer,
    rank_true_actions,
    score_actions,
)
from vjepa_byor.predictor import LatentDataset, encode_dataset

TOKENS, DIM = 6, 4


def basket(n: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    current = torch.randn(TOKENS, DIM, generator=g)
    predicted = current + 0.5 * torch.randn(n, TOKENS, DIM, generator=g)
    return predicted, current


@pytest.mark.parametrize("scorer", [L1Scorer(), ChangeWeightedL1Scorer(), ChangeWeightedL1Scorer(alpha=0.0)])
def test_candidate_equal_to_the_goal_scores_zero_and_ranks_first(scorer):
    predicted, current = basket()
    goal = predicted[3].clone()
    errors = scorer(predicted, goal, current)
    assert errors.shape == (5,)
    assert errors[3] == 0
    assert int(errors.argmin()) == 3
    assert isinstance(scorer, Scorer)


@pytest.mark.parametrize("scorer", [L1Scorer(), ChangeWeightedL1Scorer()])
def test_same_inputs_give_the_same_errors(scorer):
    predicted, current = basket()
    goal = current + 1
    first = scorer(predicted, goal, current)
    assert torch.equal(first, scorer(predicted.clone(), goal.clone(), current.clone()))


def test_l1_is_the_mean_absolute_error():
    predicted, current = basket()
    goal = torch.zeros(TOKENS, DIM)
    assert torch.allclose(L1Scorer()(predicted, goal, current), predicted.abs().mean(dim=(1, 2)))


def test_alpha_zero_is_l1():
    predicted, current = basket()
    goal = current + torch.randn(TOKENS, DIM)
    assert torch.allclose(ChangeWeightedL1Scorer(alpha=0.0)(predicted, goal, current), L1Scorer()(predicted, goal, current))


def test_change_weights_favour_tokens_that_change_and_sum_to_one():
    current = torch.zeros(TOKENS, DIM)
    goal = current.clone()
    goal[2] = 1.0  # only token 2 changes
    w = ChangeWeightedL1Scorer(alpha=0.9).weights(goal, current)
    assert torch.isclose(w.sum(), torch.tensor(1.0))
    assert int(w.argmax()) == 2 and w[2] > 0.9
    # a candidate that gets the changing token right beats one that gets a static token wrong by the same amount
    right_change = goal.clone()
    right_change[0] += 0.5
    wrong_change = goal.clone()
    wrong_change[2] -= 0.5
    errors = ChangeWeightedL1Scorer()(torch.stack([wrong_change, right_change]), goal, current)
    assert errors[1] < errors[0]


def test_no_change_means_uniform_weights():
    current = torch.randn(TOKENS, DIM)
    w = ChangeWeightedL1Scorer().weights(current, current)
    assert torch.allclose(w, torch.full((TOKENS,), 1 / TOKENS))


def test_shape_mismatch_is_rejected():
    predicted, current = basket()
    with pytest.raises(ValueError, match="expected predicted"):
        L1Scorer()(predicted, current[:-1], current)


def test_alpha_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="alpha"):
        ChangeWeightedL1Scorer(alpha=1.5)


def test_make_scorer_by_name():
    assert isinstance(make_scorer("l1"), L1Scorer)
    assert make_scorer("change-weighted", alpha=0.5).alpha == 0.5
    with pytest.raises(ValueError, match="unknown scorer"):
        make_scorer("l3")


def test_action_scores_order_rank_and_ties():
    scores = ActionScores(torch.tensor([0.3, 0.1, 0.1, 0.5]), ["a", "b", "c", "d"])
    assert scores.order().tolist() == [1, 2, 0, 3]  # tie between b and c keeps their order
    assert (scores.best, scores.best_label, scores.best_error) == (1, "b", pytest.approx(0.1))
    assert [scores.rank(i) for i in range(4)] == [3, 1, 2, 4]
    assert scores.ranked()[0] == ("b", pytest.approx(0.1))
    with pytest.raises(ValueError, match="labels"):
        ActionScores(torch.zeros(2), ["a"])


def test_score_actions_returns_cpu_float_errors():
    predicted, current = basket()
    scores = score_actions(L1Scorer(), predicted.double(), predicted[0], current, list("abcde"))
    assert scores.errors.dtype == torch.float32 and scores.errors.device.type == "cpu"
    assert scores.best_label == "a"


def test_direction_of_labels():
    assert direction_of("NE 5mm") == "NE"
    assert direction_of("forward") == "forward"


def test_rank_true_actions_reports_every_moved_sample(grid_dataset, tiny_encoder, tiny_predictor_factory, grid_mapping):
    encode_dataset(grid_dataset, tiny_encoder, "tiny")
    dataset = LatentDataset(grid_dataset, grid_mapping, "test")
    space = DiscreteActionSpace.compass([1, 3])  # the grid rig's actions
    reports = rank_true_actions(
        tiny_predictor_factory(), dataset, space.actions(), space.labels(), [L1Scorer(), ChangeWeightedL1Scorer()]
    )
    moved = int(dataset.changed.sum())
    assert [r.scorer for r in reports] == ["l1", "change-weighted"]
    for r in reports:
        assert r.samples == moved
        assert 1 <= r.mean_rank <= len(space)
        assert r.top1 <= r.top3 <= r.samples
        assert sum(len(v) for v in r.by_action.values()) == moved
        assert set(r.summary()) == {"scorer", "samples", "top1", "top3", "same_direction", "mean_rank"}
        assert "ranked 1st" in r.format(len(space))
