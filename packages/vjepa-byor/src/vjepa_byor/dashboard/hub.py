"""The dashboard's event hub: fans events out to browsers, records each run, and replays recordings.

Events are JSON-able dicts with a ``type``:

- ``camera`` / ``goal_image``: ``{"src": "data:image/png;base64,..."}``
- ``state``: ``{"state": {...}}``, the robot's position
- ``start``, ``pending``, ``step``, ``end``: one planner run, as published by the ``/reach_goal`` server
  (``pending``: in step-by-step mode, a decision waiting for "next" before its move)
- ``mode``: ``{"mode": "live"}`` or ``{"mode": "replay", "run": name}``, sent by the hub
- ``notice``: ``{"text": ..., "level": "info" | "error"}``, a message for the person driving the demo
- ``clear``: forget the last run (sent after a reset)

A browser that connects (or reloads) first gets a snapshot: the mode, the latest photo, goal photo
and state, and every event of the current (or last) run. Each run, from ``start`` to ``end``, is
saved with its photos and timing as ``<runs_dir>/<time>_<outcome>.jsonl``; :meth:`Hub.replay` plays
one back exactly as it looked live, which is also the fallback if live inference fails on stage.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Event = dict[str, Any]
_SAFE = re.compile(r"[^a-z0-9_-]+")
MAX_GAP_S = 2.0  # longest pause in a replay, before the speed factor


class Hub:
    def __init__(self, runs_dir: Path | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.runs_dir = runs_dir
        self.clock, self.sleep = clock, sleep
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[Event | None]] = []
        self._latest: dict[str, Event] = {}  # camera, goal_image, state (live)
        self._run: list[Event] = []  # events of the current or last live run
        self._recording: list[tuple[float, Event]] | None = None
        self._t0 = 0.0
        self._replay: tuple[str, threading.Event] | None = None
        self._replay_view: dict[str, Event] = {}
        self._replay_run: list[Event] = []

    # Browsers

    def subscribe(self) -> queue.Queue[Event | None]:
        """A queue that receives the snapshot now and every later event; ``None`` means the hub closed."""
        q: queue.Queue[Event | None] = queue.Queue()
        with self._lock:
            for event in self._snapshot():
                q.put(event)
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def close(self) -> None:
        self.stop_replay()
        with self._lock:
            for q in self._clients:
                q.put(None)
            self._clients.clear()

    def _snapshot(self) -> list[Event]:
        if self._replay is not None:
            view, run, mode = self._replay_view, self._replay_run, {"type": "mode", "mode": "replay",
                                                                      "run": self._replay[0]}
        else:
            view, run, mode = self._latest, self._run, {"type": "mode", "mode": "live"}
        return [mode, *(view[k] for k in ("camera", "goal_image") if k in view), *run,
                *([view["state"]] if "state" in view else [])]

    def _send(self, event: Event) -> None:
        for q in self._clients:
            q.put(event)

    # Live events

    @property
    def running(self) -> bool:
        """Whether a live run has started and not ended."""
        return self._recording is not None

    @property
    def replaying(self) -> bool:
        return self._replay is not None

    def publish(self, event: Event) -> None:
        """A live event (from ROS). Shown unless a replay is on; always recorded."""
        kind = event["type"]
        finished = None
        if kind == "start":
            self.stop_replay()  # a new live run always shows
        with self._lock:
            if kind in ("camera", "goal_image", "state"):
                self._latest[kind] = event
            if kind == "start":
                self._run = [event]
                self._t0 = self.clock()
                self._recording = [(0.0, e) for e in (self._latest.get("camera"), self._latest.get("goal_image"))
                                   if e is not None]
            elif kind in ("pending", "step", "end") and self._recording is not None:
                self._run.append(event)
            if self._recording is not None and kind != "state":
                self._recording.append((self.clock() - self._t0, event))
            if kind == "end" and self._recording is not None:
                finished, self._recording = self._recording, None
            if self._replay is None:
                self._send(event)
        if finished is not None:
            self._save(finished)

    def clear(self) -> None:
        """Forget the last run (e.g. after the robot was reset), so the page shows a fresh start."""
        with self._lock:
            if self._recording is not None:
                return  # never in the middle of a run
            self._run = []
            self._latest.pop("goal_image", None)
            if self._replay is None:
                self._send({"type": "clear"})

    def notice(self, text: str, level: str = "info") -> None:
        with self._lock:
            self._send({"type": "notice", "text": text, "level": level})

    # Recordings

    def _save(self, recording: list[tuple[float, Event]]) -> str | None:
        if self.runs_dir is None:
            return None
        end = recording[-1][1]
        outcome = _SAFE.sub("-", str(end.get("reason", "end")).lower()).strip("-")[:30] or "end"
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{outcome}"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{name}.jsonl"
        n = 1
        while path.exists():
            n += 1
            path = self.runs_dir / f"{name}-{n}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for t, event in recording:
                f.write(json.dumps({"t": round(t, 3), "event": event}) + "\n")
        return path.stem

    def runs(self) -> list[str]:
        """Saved runs, newest first."""
        if self.runs_dir is None or not self.runs_dir.is_dir():
            return []
        return sorted((p.stem for p in self.runs_dir.glob("*.jsonl")), reverse=True)

    def load(self, name: str) -> list[tuple[float, Event]]:
        if self.runs_dir is None or name not in self.runs():
            raise KeyError(f"no saved run {name!r}")
        lines = (self.runs_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        return [(float(d["t"]), d["event"]) for d in map(json.loads, lines) if d]

    # Replay

    def replay(self, name: str, speed: float = 1.0) -> None:
        """Plays a saved run back to the browsers (in a thread); live events are held back meanwhile."""
        if speed <= 0:
            raise ValueError("speed must be positive")
        recording = self.load(name)
        if self.running:
            raise RuntimeError("a live run is going on")
        self.stop_replay()
        stop = threading.Event()
        with self._lock:
            self._replay = (name, stop)
            self._replay_view, self._replay_run = {}, []
            self._send({"type": "mode", "mode": "replay", "run": name})
        threading.Thread(target=self._play, args=(name, recording, speed, stop), daemon=True).start()

    def _play(self, name: str, recording: list[tuple[float, Event]], speed: float, stop: threading.Event) -> None:
        last = 0.0
        for t, event in recording:
            gap = min(t - last, MAX_GAP_S) / speed
            last = t
            if gap > 0 and stop.wait(gap):
                return
            with self._lock:
                if stop.is_set():
                    return
                if event["type"] in ("camera", "goal_image"):
                    self._replay_view[event["type"]] = event
                elif event["type"] in ("start", "pending", "step", "end"):
                    if event["type"] == "start":
                        self._replay_run = []
                    self._replay_run.append(event)
                self._send(event)
        # stay on the last frame until the next live run or stop_replay()

    def stop_replay(self) -> None:
        """Back to live: browsers get the live snapshot again."""
        with self._lock:
            if self._replay is None:
                return
            self._replay[1].set()
            self._replay = None
            for event in self._snapshot():
                self._send(event)
