import random

import pytest
import torch

from vjepa_byor.data import ActionPolicy, EdgeAwareRandomWalk

STEPS = (-5, -1, 1, 5)  # the toy rig's actions
LOW, HIGH = 0, 20


@pytest.fixture
def State(line_rig):
    return line_rig.state_type


def walk(policy, State, steps, *, start=10, seed=0):
    """Run the policy against the toy rail; returns (chosen indices, rejection count, positions)."""
    rng = random.Random(seed)
    x, chosen, rejected, positions = start, [], [], []
    for _ in range(steps):
        state = State(x)
        index = policy.choose(state, rng)
        chosen.append(index)
        if LOW <= x + STEPS[index] <= HIGH:
            x += STEPS[index]
            policy.observe(state, index, State(x))
            rejected.append(False)
        else:
            policy.observe(state, index, None)
            rejected.append(True)
        positions.append(x)
    return chosen, rejected, positions


def make_policy(line_rig, **kwargs):
    return EdgeAwareRandomWalk(line_rig.action_space.actions(), **kwargs)


def test_satisfies_the_protocol(line_rig):
    assert isinstance(make_policy(line_rig), ActionPolicy)


def test_stays_inside_the_bounds_once_it_knows_each_action(line_rig, State):
    policy = make_policy(line_rig, bounds=(torch.tensor([LOW]), torch.tensor([HIGH])), reject_rate=0)
    chosen, rejected, _ = walk(policy, State, 3000)
    accepted = list(zip(chosen, rejected, strict=True))
    first_known = max(accepted.index((a, False)) for a in range(len(STEPS)))
    assert not any(rejected[first_known + 1 :])


def test_without_bounds_it_does_not_repeat_a_rejected_move_from_the_same_state(line_rig, State):
    policy = make_policy(line_rig, reject_rate=0)
    chosen, rejected, positions = walk(policy, State, 3000)
    seen = set()
    for i, was_rejected in enumerate(rejected):
        if was_rejected:
            key = (positions[i], chosen[i])
            assert key not in seen
            seen.add(key)


def test_deliberate_rejections_happen_at_about_the_configured_rate(line_rig, State):
    bounds = (torch.tensor([LOW]), torch.tensor([HIGH]))
    _, rejected, _ = walk(make_policy(line_rig, bounds=bounds, reject_rate=0.2), State, 4000)
    _, never, _ = walk(make_policy(line_rig, bounds=bounds, reject_rate=0.0), State, 4000)
    assert sum(rejected) > 5 * sum(never)


def test_reproducible_with_a_seed(line_rig, State):
    runs = [walk(make_policy(line_rig), State, 500, seed=7)[0] for _ in range(2)]
    assert runs[0] == runs[1]
    assert walk(make_policy(line_rig), State, 500, seed=8)[0] != runs[0]


def test_steers_back_from_an_edge(line_rig, State):
    policy = make_policy(line_rig, bounds=(torch.tensor([LOW]), torch.tensor([HIGH])), reject_rate=0)
    walk(policy, State, 200)  # learn what each action does
    rng = random.Random(0)
    picks = [policy.choose(State(1), rng) for _ in range(2000)]
    toward_centre = sum(STEPS[i] > 0 for i in picks) / len(picks)
    assert toward_centre > 0.85  # uniform over the allowed moves would give 2/3


def test_covers_the_whole_vocabulary(line_rig, State):
    policy = make_policy(line_rig, bounds=(torch.tensor([LOW]), torch.tensor([HIGH])))
    chosen, _, positions = walk(policy, State, 2000)
    assert set(chosen) == set(range(len(STEPS)))
    assert min(positions) <= 2 and max(positions) >= 18


def test_bad_bounds_are_refused(line_rig):
    with pytest.raises(ValueError):
        make_policy(line_rig, bounds=(torch.tensor([5.0]), torch.tensor([1.0])))
