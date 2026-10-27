"""Drives any rig through episodes of random actions and records every transition.

Nothing here is specific to one rig: it uses only the public interfaces in
:mod:`vjepa_byor.interfaces` and the :class:`~vjepa_byor.rig.Rig` bundle.

The interfaces have no "home" and no "read the state without moving", so the
collector learns where the rig is from the state it reports after a move. At
the start of a run, and whenever a failure leaves the state unknown, it makes
one unrecorded *sync move* first. A timed-out move is never resent: the rig may
have moved and only the reply been lost, so resending could move it twice.
"""

from __future__ import annotations

import hashlib
import json
import random
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from vjepa_byor.data.policy import ActionPolicy, EdgeAwareRandomWalk
from vjepa_byor.data.sample import (
    STATUS_OK,
    STATUS_REJECTED,
    DatasetMeta,
    Sample,
    episode_name,
)
from vjepa_byor.data.storage import DatasetWriter, next_episode_number, now, read_samples
from vjepa_byor.interfaces import Camera, CodecError, RigError, RigState
from vjepa_byor.rig import Rig

# A move that raised one of these may or may not have happened.
_MOVE_FAILURES = (OSError, CodecError)  # OSError covers TimeoutError and ConnectionError


@dataclass
class CollectorConfig:
    num_samples: int = 1000
    """Target size of the whole dataset, including samples from earlier runs."""
    steps_per_episode: int = 20
    seed: int = 0
    settle_time: float = 0.5
    """Seconds to wait after a move before the photo, so vibration and motion blur are gone."""
    reply_timeout: float | None = None
    """Overrides ``Rig.reply_timeout`` when set."""
    camera_retries: int = 2
    sync_attempts: int = 10
    max_failed_episodes: int = 3
    """Stop after this many episodes in a row end in a failure."""
    drain_timeout: float = 0.2
    """How long to wait for stale replies after a failed move."""


@dataclass
class CollectionSummary:
    target: int
    existing: int
    samples: int = 0
    episodes: int = 0
    changed: int = 0
    rejected: int = 0
    failed_moves: int = 0
    camera_failures: int = 0
    elapsed: float = 0.0
    stopped: str | None = None

    @property
    def total(self) -> int:
        return self.existing + self.samples

    @property
    def no_change_fraction(self) -> float:
        return (self.samples - self.changed) / self.samples if self.samples else 0.0

    @property
    def eta(self) -> float | None:
        """Seconds left at the current rate, or ``None`` before the first sample."""
        if not self.samples:
            return None
        return self.elapsed / self.samples * max(self.target - self.total, 0)

    def format(self) -> str:
        eta = "?" if self.eta is None else f"{self.eta:.0f}s"
        return (
            f"{self.total}/{self.target} samples ({self.samples} new, {self.episodes} episodes) | "
            f"no change {self.no_change_fraction:.0%} | rejected {self.rejected} | "
            f"failed moves {self.failed_moves} | camera failures {self.camera_failures} | eta {eta}"
        )


class Collector:
    """Records (before photo, state, action, after photo) samples from one rig.

    ``camera`` defaults to ``rig.camera``. ``sleep`` is injectable so tests run
    without waiting.
    """

    def __init__(
        self,
        rig: Rig,
        policy: ActionPolicy,
        writer: DatasetWriter,
        config: CollectorConfig | None = None,
        *,
        camera: Camera | None = None,
        sleep: Callable[[float], None] = time.sleep,
        on_progress: Callable[[CollectionSummary], None] | None = None,
    ) -> None:
        camera = camera if camera is not None else rig.camera
        if camera is None:
            raise ValueError(f"rig {rig.name!r} has no camera; pass one to the collector")
        self.rig = rig
        self.camera = camera
        self.policy = policy
        self.writer = writer
        self.config = config or CollectorConfig()
        self.sleep = sleep
        self.on_progress = on_progress
        self.actions = rig.action_space.actions()
        self.labels = rig.action_space.labels()
        self.state: RigState | None = None
        self.summary = CollectionSummary(target=self.config.num_samples, existing=0)
        self._stop: str | None = None
        if self.config.reply_timeout is not None:
            rig.reply_timeout = self.config.reply_timeout

    def request_stop(self, reason: str = "stop requested") -> None:
        """Finish the step in flight, then stop."""
        self._stop = reason

    def run(self, *, handle_signals: bool = True) -> CollectionSummary:
        root = self.writer.root
        self.summary = CollectionSummary(target=self.config.num_samples, existing=len(read_samples(root)))
        number = next_episode_number(root)
        started = time.monotonic()
        failed_in_a_row = 0
        opened_here = not self.rig.transport.is_open
        signals = self._stop_on_signals() if handle_signals else nullcontext()
        try:
            with signals:
                if opened_here:
                    self.rig.open()
                while self.summary.total < self.config.num_samples and self._stop is None:
                    if failed_in_a_row >= self.config.max_failed_episodes:
                        self._stop = f"{failed_in_a_row} episodes in a row failed"
                        break
                    self.summary.episodes += 1
                    completed = self._episode(episode_name(number), started)
                    failed_in_a_row = 0 if completed else failed_in_a_row + 1
                    number += 1
        finally:
            if opened_here:
                self.rig.close()
            self.summary.elapsed = time.monotonic() - started
            self.summary.stopped = self._stop
            self.writer.log_event(
                "run_finished",
                total=self.summary.total,
                new=self.summary.samples,
                stopped=self._stop,
            )
        return self.summary

    # One episode

    def _episode(self, episode_id: str, started: float) -> bool:
        """Record one episode; returns ``False`` if it ended in a failure."""
        rng = random.Random(f"{self.config.seed}:{episode_id}")
        if self.state is None:
            self.state = self._sync(rng, episode_id)
            if self.state is None:
                return False
        state = self.state

        self.sleep(self.config.settle_time)
        image = self._capture(episode_id, 0)
        if image is None:
            return False
        before = self.writer.write_frame(episode_id, 0, image)

        for step in range(self.config.steps_per_episode):
            if self._stop is not None or self.summary.total >= self.config.num_samples:
                break
            index = self.policy.choose(state, rng)
            action = self.actions[index]
            command = self.rig.codec.encode_action(action)
            status, error = STATUS_OK, None
            try:
                after_state = self.rig.execute(action)
            except RigError as e:
                after_state, status, error = state, STATUS_REJECTED, str(e)
                self.writer.log_event(
                    "rejected", episode_id=episode_id, step_index=step, command=command, error=error
                )
            except _MOVE_FAILURES as e:
                self.summary.failed_moves += 1
                self.writer.log_event(
                    "move_failed",
                    episode_id=episode_id,
                    step_index=step,
                    command=command,
                    error=repr(e),
                    action="episode ended; state unknown until the next sync move",
                )
                self.state = None
                self._drain()
                return False
            self.policy.observe(state, index, after_state if status == STATUS_OK else None)

            self.sleep(self.config.settle_time)
            image = self._capture(episode_id, step + 1)
            if image is None:
                self.state = after_state
                return False
            after = self.writer.write_frame(episode_id, step + 1, image)

            changed = after_state.to_dict() != state.to_dict()
            self.writer.append_sample(
                Sample(
                    sample_id=f"{episode_id}_s{step:04d}",
                    episode_id=episode_id,
                    step_index=step,
                    before_frame=before,
                    after_frame=after,
                    state_before=state.to_dict(),
                    state_after=after_state.to_dict(),
                    state_before_tensor=state.as_tensor().tolist(),
                    state_after_tensor=after_state.as_tensor().tolist(),
                    action_index=index,
                    action_name=self.labels[index],
                    action=action.tolist(),
                    command=command,
                    status=status,
                    changed=changed,
                    resolution=list(self.camera.resolution),
                    timestamp=now(),
                    error=error,
                )
            )
            self.summary.samples += 1
            self.summary.changed += changed
            self.summary.rejected += status == STATUS_REJECTED
            self.summary.elapsed = time.monotonic() - started
            if self.on_progress is not None:
                self.on_progress(self.summary)
            state = self.state = after_state
            before = after
        return True

    # Helpers

    def _sync(self, rng: random.Random, episode_id: str) -> RigState | None:
        """Make unrecorded moves until the rig reports where it is."""
        self._drain()
        untried = list(range(len(self.actions)))
        for attempt in range(self.config.sync_attempts):
            index = rng.choice(untried or range(len(self.actions)))
            if index in untried:
                untried.remove(index)
            try:
                state = self.rig.execute(self.actions[index])
            except RigError as e:
                self.writer.log_event("sync_rejected", episode_id=episode_id, attempt=attempt, error=str(e))
                continue
            except _MOVE_FAILURES as e:
                self.summary.failed_moves += 1
                self.writer.log_event("sync_failed", episode_id=episode_id, attempt=attempt, error=repr(e))
                self._drain()
                continue
            self.writer.log_event("synced", episode_id=episode_id, state=state.to_dict())
            return state
        self.writer.log_event("sync_gave_up", episode_id=episode_id, attempts=self.config.sync_attempts)
        return None

    def _drain(self) -> None:
        """Discard replies that arrived late, so they are not mistaken for the next one."""
        stale = []
        for _ in range(100):
            try:
                stale.append(self.rig.transport.receive(self.config.drain_timeout))
            except Exception:
                break
        if stale:
            self.writer.log_event("stale_replies_drained", lines=stale)

    def _capture(self, episode_id: str, frame: int) -> torch.Tensor | None:
        expected = (3, *self.camera.resolution)
        for attempt in range(self.config.camera_retries + 1):
            try:
                image = self.camera.capture()
                if image.dtype != torch.uint8 or tuple(image.shape) != expected:
                    raise ValueError(f"camera returned {image.dtype} {tuple(image.shape)}, expected uint8 {expected}")
                return image
            except Exception as e:
                self.summary.camera_failures += 1
                self.writer.log_event(
                    "camera_failed", episode_id=episode_id, frame=frame, attempt=attempt, error=repr(e)
                )
        self.writer.log_event("episode_ended", episode_id=episode_id, reason="camera failed")
        return None

    @contextmanager
    def _stop_on_signals(self) -> Iterator[None]:
        """First Ctrl-C or SIGTERM: finish the current step and stop. Second Ctrl-C: abort."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return

        def handler(signum: int, frame: Any) -> None:
            if self._stop is not None and signum == signal.SIGINT:
                raise KeyboardInterrupt
            self._stop = f"received {signal.Signals(signum).name}"

        previous = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield
        finally:
            for s, h in previous.items():
                signal.signal(s, h)


# Building a run


def _collector_version() -> str:
    try:
        return version("vjepa-byor")
    except PackageNotFoundError:
        return "unknown"


def config_hash(options: dict[str, Any] | None) -> str:
    canonical = json.dumps(options or {}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_meta(rig: Rig, camera: Camera, rig_options: dict[str, Any] | None = None) -> DatasetMeta:
    return DatasetMeta(
        rig=rig.name,
        state_type=f"{rig.state_type.__module__}.{rig.state_type.__qualname__}",
        action_labels=rig.action_space.labels(),
        actions=rig.action_space.actions().tolist(),
        resolution=list(camera.resolution),
        rig_config_hash=config_hash(rig_options),
        collector_version=_collector_version(),
        created_at=now(),
    )


@dataclass
class CollectRun:
    """Everything :func:`collect` needs beyond the rig and the output directory."""

    config: CollectorConfig = field(default_factory=CollectorConfig)
    camera: Camera | None = None
    policy: ActionPolicy | None = None
    resume: bool = False
    rig_options: dict[str, Any] | None = None
    on_progress: Callable[[CollectionSummary], None] | None = None
    sleep: Callable[[float], None] = time.sleep
    handle_signals: bool = True


def collect(rig: Rig, out: str | Path, run: CollectRun | None = None) -> CollectionSummary:
    """Collect into ``out`` until it holds ``run.config.num_samples`` samples.

    An existing dataset is only added to when ``run.resume`` is set, and only
    if it was collected with the same rig, vocabulary and resolution.
    """
    run = run or CollectRun()
    camera = run.camera if run.camera is not None else rig.camera
    if camera is None:
        raise ValueError(f"rig {rig.name!r} has no camera; pass one")
    meta = build_meta(rig, camera, run.rig_options)
    out = Path(out)
    if (out / "meta.json").exists():
        if not run.resume:
            raise FileExistsError(f"{out} already holds a dataset; pass resume=True (--resume) to add to it")
        writer = DatasetWriter.resume(out, meta)
    else:
        writer = DatasetWriter.create(out, meta)
    policy = run.policy or EdgeAwareRandomWalk(rig.action_space.actions())
    with writer:
        collector = Collector(
            rig, policy, writer, run.config, camera=camera, sleep=run.sleep, on_progress=run.on_progress
        )
        return collector.run(handle_signals=run.handle_signals)
