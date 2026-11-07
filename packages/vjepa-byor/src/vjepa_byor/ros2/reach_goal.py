"""``/reach_goal``: the planner as a ROS 2 action server.

    ros2 action send_goal --feedback /reach_goal vjepa_byor_interfaces/action/ReachGoal "{goal_state: [120, 60]}"

The action (``vjepa_byor_interfaces/action/ReachGoal``, built from ``ros2/`` in the repository) takes
a goal photo or, with a ``render_goal`` function, a goal rig state. Every decision goes out as
feedback (all candidate moves with their predicted errors, the move made); the result says whether
the planner reached the goal. A cancel stops the run before the next move. One goal at a time: a
goal sent while another runs is rejected.

The goal photo is also published on ``<action>/goal_image`` (latched), for the dashboard and
``rqt_image_view``, and every run goes out as JSON events on ``<action>/events``
(``std_msgs/String``): ``start`` (goal, start state, the moves as ``labels`` and ``actions``),
one ``step`` per decision (a :class:`PlanStep` as a dict) and ``end`` (the result and the goal
status), whoever sent the goal. The dashboard draws and records runs from these.

Step-by-step mode (``step_by_step: true`` in the goal): after scoring every move and before making
the best one, the server publishes a ``pending`` event (the decision, with the ``planned`` move) and
waits for a call to ``<action>/next`` (``std_srvs/Trigger``), so an audience can look at what the
model pictured before the robot moves. A cancel ends the wait. Needs ``rclpy`` and the interfaces package, so import it only in a ROS 2
environment. ``apps/ros2/reach_goal_server.py`` runs it with the fine-tuned model.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace

import torch
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vjepa_byor_interfaces.action import ReachGoal

from vjepa_byor.planning import Planner, PlannerConfig, PlanResult, PlanStep, Scorer
from vjepa_byor.rig import Rig
from vjepa_byor.ros2.link import image_to_tensor, tensor_to_image

ACTION_NAME = "/reach_goal"
GoalRenderer = Callable[[Sequence[float]], torch.Tensor]
"""Rig state -> the photo the camera would take there, ``uint8 (3, H, W)``; raises ``ValueError`` if unreachable."""

LATCHED = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class GoalError(ValueError):
    """The goal can't be used: both or neither given, wrong size, or out of reach."""


class ReachGoalServer(Node):
    """Serves ``/reach_goal`` for ``rig`` (a ``ros2`` rig, opened on the first goal if not yet open)."""

    def __init__(
        self,
        model,
        rig: Rig,
        config: PlannerConfig | None = None,
        scorer: Scorer | None = None,
        render_goal: GoalRenderer | None = None,
        action_name: str = ACTION_NAME,
        **node_kwargs,
    ) -> None:
        super().__init__("reach_goal_server", **node_kwargs)
        if not hasattr(rig.transport, "current_state"):
            raise TypeError(f"rig {rig.name!r} can't report its state without moving; use the ros2 rig")
        self.model, self.rig, self.scorer, self.render_goal = model, rig, scorer, render_goal
        self.config = config or PlannerConfig()
        self.state_names = list(rig.state_type.field_names())
        self._busy = threading.Lock()
        self._next = threading.Event()
        self._waiting = False
        self.goal_image_pub = self.create_publisher(Image, f"{action_name}/goal_image", LATCHED)
        self.events_pub = self.create_publisher(String, f"{action_name}/events", 50)
        self._server = ActionServer(
            self, ReachGoal, action_name,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _handle: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_service(Trigger, f"{action_name}/next", self._on_next, callback_group=ReentrantCallbackGroup())
        self.get_logger().info(
            f"{action_name} ready: rig state {self.state_names}, {len(rig.action_space)} moves, "
            f"goals as {'a state or ' if render_goal else ''}a photo"
        )

    def destroy_node(self) -> None:
        self._server.destroy()
        super().destroy_node()

    # Goals

    def _on_goal(self, _request: ReachGoal.Goal) -> GoalResponse:
        if not self._busy.acquire(blocking=False):
            self.get_logger().warning("rejected a goal: another goal is running")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT  # _execute releases the lock

    def goal_photo(self, request: ReachGoal.Goal) -> tuple[torch.Tensor, list[float] | None]:
        """The goal photo, and the goal state if one was given. Raises :class:`GoalError`."""
        has_state, has_image = len(request.goal_state) > 0, len(request.goal_image.data) > 0
        if has_state == has_image:
            raise GoalError("give exactly one of goal_state and goal_image")
        if has_image:
            try:
                return image_to_tensor(request.goal_image, self.rig.camera.resolution), None
            except (ValueError, RuntimeError) as e:
                raise GoalError(f"goal_image: {e}") from e
        state = [float(v) for v in request.goal_state]
        if len(state) != len(self.state_names):
            raise GoalError(f"goal_state needs {len(self.state_names)} values {self.state_names}, got {len(state)}")
        if self.render_goal is None:
            raise GoalError("this server has no goal renderer: send a goal_image")
        try:
            return self.render_goal(state), state
        except ValueError as e:
            raise GoalError(f"goal_state {state}: {e}") from e

    def _on_next(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = self._waiting
        response.message = "moving" if self._waiting else "no step-by-step run is waiting for a move"
        if self._waiting:
            self._next.set()
        return response

    def _wait_for_next(self, handle: ServerGoalHandle, step: PlanStep, best: int) -> bool:
        """Step-by-step mode: show the decision, then hold the move until ``next`` (``False`` if cancelled)."""
        self._next.clear()
        self._waiting = True
        self.publish_event("pending", **step.to_dict(), planned=step.labels[best])
        try:
            while not self._next.wait(0.05):
                if handle.is_cancel_requested or not handle.is_active:
                    return False
            return True
        finally:
            self._waiting = False

    def publish_event(self, kind: str, **data) -> None:
        self.events_pub.publish(String(data=json.dumps({"type": kind, **data})))

    # The run

    def _execute(self, handle: ServerGoalHandle) -> ReachGoal.Result:
        try:
            return self._run(handle)
        finally:
            self._busy.release()

    def _run(self, handle: ServerGoalHandle) -> ReachGoal.Result:
        log = self.get_logger()
        request = handle.request
        try:
            photo, goal_state = self.goal_photo(request)
            self.rig.open()
            start = self.rig.transport.current_state()
        except (GoalError, ConnectionError) as e:
            log.warning(f"aborted: {e}")
            handle.abort()
            return self._end(self._result(None, None, reason=f"error: {e}"), "aborted")

        self.goal_image_pub.publish(tensor_to_image(photo, self.get_clock().now().to_msg(), "goal"))
        config = replace(self.config, max_steps=request.max_steps) if request.max_steps else self.config
        target = f"state {goal_state}" if goal_state is not None else "a goal photo"
        mode = ", step by step" if request.step_by_step else ""
        log.info(f"goal: {target} from {start.to_dict()}, at most {config.max_steps} moves{mode}")
        self.publish_event(
            "start", goal_state=goal_state, start=start.to_dict(), state_names=self.state_names,
            max_steps=config.max_steps, threshold=config.threshold, step_by_step=request.step_by_step,
            labels=self.rig.action_space.labels(), actions=self.rig.action_space.actions().tolist(),
        )

        planner = Planner(self.model, self.rig, config, self.scorer)
        try:
            result = planner.run(
                photo, start,
                on_step=lambda step: self._on_step(handle, step),
                should_stop=lambda: handle.is_cancel_requested,
                before_move=(lambda step, best: self._wait_for_next(handle, step, best))
                if request.step_by_step else None,
            )
        except (TimeoutError, ConnectionError, OSError) as e:
            log.error(f"aborted: the robot stopped answering: {e!r}")
            handle.abort()
            return self._end(self._result(None, goal_state, reason=f"error: {e}"), "aborted")

        log.info(f"{result.reason}: {result.moves} moves, at {result.final_state}, error {result.final_error:.4f}")
        if result.reason == "stopped":
            handle.canceled()
            status = "canceled"
        elif result.reached:
            handle.succeed()
            status = "succeeded"
        else:
            handle.abort()
            status = "aborted"
        return self._end(self._result(result, goal_state), status)

    def _on_step(self, handle: ServerGoalHandle, step: PlanStep) -> None:
        handle.publish_feedback(self._feedback(step))
        self.publish_event("step", **step.to_dict())

    def _end(self, result: ReachGoal.Result, status: str) -> ReachGoal.Result:
        self.publish_event(
            "end", status=status, reached=result.reached, reason=result.reason,
            final_state=dict(zip(self.state_names, result.final_state)), moves=result.moves,
            final_error=result.final_error,
            distance_to_goal=None if result.distance_to_goal < 0 else result.distance_to_goal,
        )
        return result

    def _values(self, state: dict | None) -> list[float]:
        return [] if state is None else [float(state[n]) for n in self.state_names]

    def _feedback(self, step: PlanStep) -> ReachGoal.Feedback:
        return ReachGoal.Feedback(
            step=step.step,
            state_names=self.state_names,
            state=self._values(step.state),
            error=step.current_error,
            chosen=step.chosen if step.chosen is not None and step.error is None else "",
            chosen_error=step.chosen_error if step.chosen_error is not None and step.error is None else 0.0,
            refused=step.refused,
            state_after=self._values(step.state_after),
            labels=step.labels,
            errors=step.errors,
            seconds=step.seconds,
        )

    def _result(self, result: PlanResult | None, goal_state: list[float] | None, reason: str = "") -> ReachGoal.Result:
        if result is None:
            return ReachGoal.Result(reached=False, reason=reason, state_names=self.state_names, distance_to_goal=-1.0)
        final = self._values(result.final_state)
        return ReachGoal.Result(
            reached=result.reached,
            reason=result.reason,
            state_names=self.state_names,
            final_state=final,
            moves=result.moves,
            final_error=result.final_error,
            distance_to_goal=math.dist(final, goal_state) if goal_state is not None else -1.0,
        )
