"""The planning loop: score every action, execute the best, take a new photo, repeat until the goal is reached.

::

    planner = Planner(model, rig, PlannerConfig(threshold=0.16))
    result = planner.run(goal_photo, start_state, on_step=print)
    result.reached, len(result.steps)

Each step encodes the current photo, imagines the outcome of every action in
the rig's basket with the fine-tuned predictor, and scores each imagined
outcome against the goal photo's latents. The planner stops when the current
photo's own error to the goal falls below ``threshold`` (the robot is there),
when the rig comes back to a state it was already in (it is circling, usually
right next to the goal), after ``max_steps`` moves, or when ``should_stop`` says so. If the rig refuses
the best move (for example past its travel limits), the next best is tried.

With ``horizon > 1`` each first action is scored by the best sequence that
starts with it: the predictor's outputs are fed back in as the next "photo",
``horizon`` steps deep. Later steps only try ``lookahead_actions`` (by default
the whole basket), and only the ``beam`` best partial sequences are expanded
at each depth, so the cost grows linearly with the horizon. This needs to know
the rig state after an action without moving; ``state_after`` provides it, and
defaults to adding the action to the state when both have the same size (true
for rigs whose actions are relative moves of their state, like an XY stage).

Measured on the simulated stage (``checkpoints/sim-10k``, 40 random goals):
greedy reached 31/40, 2-step look-ahead 29/40 at 17x the time per decision, and
3-step look-ahead 10/40. A predictor fine-tuned on single steps drifts when fed
its own predictions, so keep ``horizon=1`` unless yours was trained on rollouts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from vjepa_byor.interfaces import RigError, RigState
from vjepa_byor.planning.scorer import ActionScores, L1Scorer, Scorer
from vjepa_byor.predictor.inference import ActionPredictor
from vjepa_byor.rig import Rig

StateUpdate = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``(states (N, state_dim), actions (N, action_dim)) -> states after the actions (N, state_dim)``."""


def additive_state_update(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """The state after a relative move: ``state + action``."""
    return states + actions


@dataclass(frozen=True)
class PlannerConfig:
    threshold: float = 0.165
    """Stop once the current photo's error to the goal is at or below this (same units as the scorer)."""
    max_steps: int = 30
    stop_on_cycle: bool = True
    """Stop when the rig is back in a state it was already in during this run: it is circling, not converging."""
    horizon: int = 1
    """How many steps to look ahead when scoring a first action; 1 is greedy."""
    lookahead_actions: Sequence[int] | None = None
    """Indices into the basket tried after the first step; ``None`` tries the whole basket."""
    beam: int = 8
    """Partial sequences kept and expanded at each look-ahead depth."""

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.horizon < 1 or self.beam < 1:
            raise ValueError("max_steps, horizon and beam must be at least 1")


@dataclass
class PlanStep:
    """One decision: where the rig was, how every candidate scored, and what was executed."""

    step: int
    state: dict[str, Any]
    current_error: float
    labels: list[str]
    errors: list[float]
    """One error per first action, in the basket's order (with look-ahead: the best sequence starting with it)."""
    chosen: str | None
    chosen_error: float | None
    seconds: float
    """Time to encode the photo and score the basket (not the move)."""
    state_after: dict[str, Any] | None = None
    refused: list[str] = field(default_factory=list)
    """Better-scoring actions the rig refused this step, before ``chosen`` was executed."""
    error: str | None = None
    """Set if every action was refused: the last refusal's message."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanResult:
    reached: bool
    reason: str
    """``reached``, ``cycle`` (back in an earlier state, usually hovering next to the goal), ``max_steps``,
    ``stopped`` or ``rig_error`` (the rig refused every action)."""
    final_state: dict[str, Any]
    final_error: float
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def moves(self) -> int:
        """Moves executed (the last step may only be the check that found the goal)."""
        return sum(s.chosen is not None and s.error is None for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {**{k: v for k, v in asdict(self).items() if k != "steps"}, "moves": self.moves,
                "steps": [s.to_dict() for s in self.steps]}


class Planner:
    """Drives ``rig`` toward a goal photo with ``model`` (encoder + fine-tuned predictor) and ``scorer``."""

    def __init__(
        self,
        model: ActionPredictor,
        rig: Rig,
        config: PlannerConfig | None = None,
        scorer: Scorer | None = None,
        state_after: StateUpdate | None = None,
    ) -> None:
        if rig.camera is None:
            raise ValueError(f"rig {rig.name!r} has no camera; the planner needs photos")
        self.model = model
        self.rig = rig
        self.config = config or PlannerConfig()
        self.scorer = scorer or L1Scorer()
        self.actions = rig.action_space.actions().float()
        self.labels = rig.action_space.labels()
        if self.config.lookahead_actions is not None:
            bad = [i for i in self.config.lookahead_actions if not 0 <= i < len(self.actions)]
            if bad:
                raise ValueError(f"lookahead_actions {bad} are outside the basket of {len(self.actions)}")
        self.state_after = state_after or additive_state_update

    # ------------------------------------------------------------------ scoring

    def score(self, z: torch.Tensor, z_goal: torch.Tensor, state: torch.Tensor) -> ActionScores:
        """One error per first action, from photo latents ``z`` and rig state tensor ``state``."""
        horizon = self.config.horizon
        if horizon > 1 and self.state_after is additive_state_update and state.shape[-1] != self.actions.shape[1]:
            raise ValueError(
                f"look-ahead (horizon > 1) needs state_after=...: this rig's states have {state.shape[-1]} values "
                f"and its actions {self.actions.shape[1]}, so the state after an action isn't state + action"
            )
        predicted = self.model.predict(z, state, self.actions)
        errors = self.scorer(predicted, z_goal, z).float().cpu()
        if self.config.horizon > 1:
            errors = self._look_ahead(predicted, errors, z_goal, z, state)
        return ActionScores(errors, list(self.labels))

    def _look_ahead(
        self, predicted: torch.Tensor, errors: torch.Tensor, z_goal: torch.Tensor, z: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Best error reachable within ``horizon`` steps for each first action (never worse than one step)."""
        later = self.actions if self.config.lookahead_actions is None else self.actions[list(self.config.lookahead_actions)]
        n, m = len(self.actions), len(later)
        best = errors.clone()
        # the frontier: every partial sequence still being expanded, as (first action, latents, state after it)
        firsts = torch.arange(n)
        latents = predicted
        states = self.state_after(state.expand(n, -1), self.actions)
        for _ in range(self.config.horizon - 1):
            f = len(firsts)
            firsts = firsts.repeat_interleave(m)
            acts = later.repeat(f, 1)
            latents = self.model.predict_pairs(latents.repeat_interleave(m, dim=0), states.repeat_interleave(m, dim=0), acts)
            states = self.state_after(states.repeat_interleave(m, dim=0), acts)
            err = self.scorer(latents, z_goal, z).float().cpu()
            best = best.scatter_reduce(0, firsts, err, reduce="amin")
            # expand the beam best sequences next, plus the best one for each first action, so that every
            # first action keeps being scored to the full depth
            keep = set(torch.argsort(err, stable=True)[: self.config.beam].tolist())
            for a in range(n):
                mine = (firsts == a).nonzero().flatten()
                keep.add(int(mine[err[mine].argmin()]))
            keep_t = torch.tensor(sorted(keep))
            firsts, states = firsts[keep_t], states[keep_t]
            latents = latents[keep_t.to(latents.device)]
        return best

    # ------------------------------------------------------------------ the loop

    def run(
        self,
        goal_photo: torch.Tensor,
        start_state: RigState,
        on_step: Callable[[PlanStep], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        before_move: Callable[[PlanStep, int], bool] | None = None,
    ) -> PlanResult:
        """Plan and move until the goal is reached, ``max_steps`` moves are made, or ``should_stop()`` is true.

        ``start_state`` is where the rig is now (the interfaces can't read it
        without moving). ``on_step`` is called after every decision, with the
        move's outcome filled in. ``before_move(step, best)`` is called after
        the basket is scored and before anything moves, with the index of the
        best-scoring action; it may block (e.g. to wait for a person: step-by-step
        mode), and returning ``False`` stops the run (reason ``stopped``).
        """
        cfg = self.config
        z_goal = self.model.encode(goal_photo)
        state = start_state
        steps: list[PlanStep] = []
        visited: set[str] = set()

        def finish(reached: bool, reason: str, err: float) -> PlanResult:
            return PlanResult(reached, reason, state.to_dict(), err, steps)

        for k in range(cfg.max_steps + 1):
            t0 = time.perf_counter()
            z = self.model.encode(self.rig.camera.capture())
            current = float(self.scorer(z.unsqueeze(0), z_goal, z)[0])
            at_goal = current <= cfg.threshold
            key = json.dumps(state.to_dict(), sort_keys=True)
            cycled = cfg.stop_on_cycle and key in visited
            visited.add(key)
            scores = None if at_goal else self.score(z, z_goal, state.as_tensor().float())
            step = PlanStep(
                step=k,
                state=state.to_dict(),
                current_error=current,
                labels=list(self.labels),
                errors=[] if scores is None else scores.errors.tolist(),
                chosen=None,
                chosen_error=None,
                seconds=time.perf_counter() - t0,
            )
            steps.append(step)
            reason = (
                "reached" if at_goal
                else "cycle" if cycled
                else "max_steps" if k == cfg.max_steps
                else "stopped" if should_stop is not None and should_stop()
                else None
            )
            if reason is None and before_move is not None:
                assert scores is not None
                if not before_move(step, int(scores.order()[0])):
                    reason = "stopped"
            if reason is None:
                assert scores is not None
                # a refused move (e.g. past the travel limits) leaves the rig where it was: try the next best
                for index in scores.order().tolist():
                    step.chosen, step.chosen_error = self.labels[index], float(scores.errors[index])
                    try:
                        state = self.rig.execute(self.actions[index])
                    except RigError as e:
                        step.refused.append(self.labels[index])
                        step.error = str(e)
                        continue
                    step.state_after, step.error = state.to_dict(), None
                    break
                else:
                    reason = "rig_error"
            if on_step is not None:
                on_step(step)
            if reason is not None:
                return finish(at_goal, reason, current)
        raise AssertionError("unreachable")
