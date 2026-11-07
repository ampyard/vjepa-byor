import itertools
import json

import pytest
import torch

from vjepa_byor import RigError
from vjepa_byor.planning import Planner, PlannerConfig
from vjepa_byor.predictor import ActionPredictor

from .conftest import HEIGHT, WIDTH, GridState, make_grid_rig


class PerfectModel:
    """Stands in for ActionPredictor: latents are the marker's position, and predictions are exact.

    Latents are one token holding (x, y) in pixels, so the L1 scorer's error is
    half the Manhattan distance to the goal, in pixels.
    """

    def __init__(self, rig) -> None:
        self.camera = rig.camera
        self.calls = 0

    def _latent(self, x: float, y: float) -> torch.Tensor:
        h, w = self.camera.resolution
        px = self.camera._pixel(x, 0, WIDTH, w)
        py = self.camera._pixel(y, 0, HEIGHT, h)
        return torch.tensor([[float(px), float(py)]])

    def encode(self, photo: torch.Tensor) -> torch.Tensor:
        ys, xs = (photo[0] > 0).nonzero(as_tuple=True)
        return torch.tensor([[float(xs.min()), float(ys.min())]])

    def predict(self, z, state, actions):
        self.calls += 1
        return torch.stack([self._latent(*(state + a).tolist()) for a in actions])

    def predict_pairs(self, z, states, actions):
        self.calls += 1
        return torch.stack([self._latent(*(s + a).tolist()) for s, a in zip(states, actions)])


def goal_photo(x: float, y: float) -> torch.Tensor:
    rig = make_grid_rig()
    rig.transport.responder.x, rig.transport.responder.y = x, y
    return rig.camera.capture()


def start_state(rig) -> GridState:
    return GridState(rig.transport.responder.x, rig.transport.responder.y)


def test_greedy_planner_reaches_the_goal_and_stops_by_itself():
    rig = make_grid_rig()
    seen = []
    with rig:
        planner = Planner(PerfectModel(rig), rig, PlannerConfig(threshold=0.0, max_steps=20))
        result = planner.run(goal_photo(17, 4), start_state(rig), on_step=seen.append)
    assert result.reached and result.reason == "reached"
    assert result.final_state == {"x": 17.0, "y": 4.0}
    assert result.final_error == 0
    assert seen == result.steps
    assert result.moves == len(result.steps) - 1 <= 8  # 7 right, 6 down with 1- and 3-steps
    last = result.steps[-1]
    assert last.chosen is None and last.errors == []
    first = result.steps[0]
    assert len(first.errors) == len(rig.action_space) and first.chosen == "SE 3"
    assert first.state == {"x": 10.0, "y": 10.0} and first.state_after == {"x": 13.0, "y": 7.0}
    json.dumps(result.to_dict())


def test_stops_after_max_steps_without_a_last_move():
    rig = make_grid_rig()
    with rig:
        result = Planner(PerfectModel(rig), rig, PlannerConfig(threshold=-1, max_steps=3)).run(
            goal_photo(17, 4), start_state(rig)
        )
    assert not result.reached and result.reason == "max_steps"
    assert result.moves == 3 and len(result.steps) == 4
    assert result.steps[-1].chosen is None and len(result.steps[-1].errors) == len(rig.action_space)


def test_should_stop_ends_the_run_before_moving():
    rig = make_grid_rig()
    with rig:
        result = Planner(PerfectModel(rig), rig).run(goal_photo(17, 4), start_state(rig), should_stop=lambda: True)
    assert result.reason == "stopped" and result.moves == 0
    assert result.final_state == {"x": 10.0, "y": 10.0}


def test_a_refused_move_falls_back_to_the_next_best(monkeypatch):
    rig = make_grid_rig()
    execute = rig.execute
    refused = {"SE 3"}

    def picky(action):
        if rig.action_space.labels()[rig.action_space.index_of(action)] in refused:
            raise RigError("out of range")
        return execute(action)

    monkeypatch.setattr(rig, "execute", picky)
    with rig:
        result = Planner(PerfectModel(rig), rig, PlannerConfig(threshold=0.0, max_steps=30)).run(
            goal_photo(17, 4), start_state(rig)
        )
    assert result.reached
    first = result.steps[0]
    assert first.refused == ["SE 3"] and first.chosen != "SE 3" and first.error is None


def test_rig_refusing_everything_ends_the_run(monkeypatch):
    rig = make_grid_rig()
    monkeypatch.setattr(rig, "execute", lambda action: (_ for _ in ()).throw(RigError("estop")))
    with rig:
        result = Planner(PerfectModel(rig), rig).run(goal_photo(17, 4), start_state(rig))
    assert result.reason == "rig_error" and result.moves == 0
    assert len(result.steps[0].refused) == len(rig.action_space) and result.steps[0].error == "estop"


def test_two_step_look_ahead_scores_each_first_action_by_its_best_continuation():
    rig = make_grid_rig()
    model = PerfectModel(rig)
    goal = (17, 4)
    planner = Planner(model, rig, PlannerConfig(horizon=2, beam=10**6))
    z_goal = model._latent(*goal)[0:1]
    state = torch.tensor([10.0, 10.0])
    z = model._latent(*state.tolist())[0:1]
    scores = planner.score(z, z_goal, state)
    actions = rig.action_space.actions()
    for i, a in enumerate(actions):
        one = planner.scorer(model._latent(*(state + a).tolist()).unsqueeze(0), z_goal, z)[0]
        two = min(planner.scorer(model._latent(*(state + a + b).tolist()).unsqueeze(0), z_goal, z)[0] for b in actions)
        assert scores.errors[i] == pytest.approx(float(min(one, two)))
    greedy = Planner(model, rig).score(z, z_goal, state)
    assert (scores.errors <= greedy.errors + 1e-6).all()


def test_look_ahead_cost_grows_linearly_with_horizon_and_beam():
    rig = make_grid_rig()
    n = len(rig.action_space)
    for horizon in (2, 3, 4):
        model = PerfectModel(rig)
        lookahead = [i for i, label in enumerate(rig.action_space.labels()) if label.endswith(" 3")]
        planner = Planner(model, rig, PlannerConfig(horizon=horizon, beam=4, lookahead_actions=lookahead))
        scores = planner.score(model._latent(10, 10)[0:1], model._latent(17, 4)[0:1], torch.tensor([10.0, 10.0]))
        assert scores.errors.shape == (n,)
        assert model.calls == horizon  # one basket call, then one batched call per extra depth


def test_look_ahead_needs_a_state_update_when_states_are_not_relative_moves():
    rig = make_grid_rig()
    planner = Planner(PerfectModel(rig), rig, PlannerConfig(horizon=2))
    with pytest.raises(ValueError, match="state_after"):
        planner.score(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([1.0, 2.0, 3.0]))


def test_config_and_lookahead_indices_are_checked():
    rig = make_grid_rig()
    with pytest.raises(ValueError):
        PlannerConfig(horizon=0)
    with pytest.raises(ValueError, match="outside the basket"):
        Planner(PerfectModel(rig), rig, PlannerConfig(lookahead_actions=[99]))


def test_runs_with_the_real_action_predictor_on_tiny_models(tiny_encoder, tiny_predictor_factory, grid_mapping):
    rig = make_grid_rig()
    model = ActionPredictor(tiny_encoder, tiny_predictor_factory(), grid_mapping)
    with rig:
        result = Planner(model, rig, PlannerConfig(max_steps=2, horizon=2, beam=3)).run(
            goal_photo(17, 4), start_state(rig)
        )
    assert len(result.steps) >= 1
    assert all(len(s.errors) == len(rig.action_space) for s in result.steps if s.chosen)
    assert list(itertools.chain.from_iterable(s.errors for s in result.steps))


def test_coming_back_to_an_earlier_state_stops_the_run():
    rig = make_grid_rig()
    with rig:
        result = Planner(PerfectModel(rig), rig, PlannerConfig(threshold=-1, max_steps=30)).run(
            goal_photo(17, 4), start_state(rig)
        )
    assert result.reason == "cycle" and not result.reached
    assert result.final_state == result.steps[-1].state
    assert result.moves < 30
    with rig:
        endless = Planner(PerfectModel(rig), rig, PlannerConfig(threshold=-1, max_steps=12, stop_on_cycle=False)).run(
            goal_photo(17, 4), GridState(rig.transport.responder.x, rig.transport.responder.y)
        )
    assert endless.reason == "max_steps" and endless.moves == 12


def test_before_move_sees_every_decision_before_the_move_and_can_stop_the_run():
    rig = make_grid_rig()
    model = PerfectModel(rig)
    seen = []

    def before_move(step, best):
        seen.append((step.step, step.state, step.chosen, step.labels[best], step.errors[best] == min(step.errors)))
        return len(seen) < 2  # let one move through, stop at the second

    with rig:
        result = Planner(model, rig, PlannerConfig(threshold=0.0)).run(
            goal_photo(17, 4), start_state(rig), before_move=before_move)
    assert result.reason == "stopped" and result.moves == 1
    assert [s[2] for s in seen] == [None, None]  # called before anything was chosen or moved
    assert all(s[4] for s in seen)  # best = lowest predicted error
    assert seen[1][1] == result.steps[0].state_after  # the second decision starts where the first move ended
