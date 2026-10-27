import re
from collections import Counter
from pathlib import Path

import pytest
import torch

from vjepa_byor.data import (
    Collector,
    CollectorConfig,
    CollectRun,
    DatasetFormatError,
    DatasetWriter,
    EdgeAwareRandomWalk,
    build_meta,
    collect,
    ensure_splits,
    export,
    load_frame,
    read_samples,
    verify,
)
from vjepa_byor.data.storage import MANIFEST, read_events
from vjepa_byor.testing import FakeTransport
from vjepa_byor.testing.fake_camera import MarkerCamera


def run(rig, root, *, resume=False, policy=None, camera=None, **config):
    config = CollectorConfig(**{"settle_time": 0, **config})
    return collect(
        rig, root, CollectRun(config=config, resume=resume, policy=policy, camera=camera, handle_signals=False)
    )


def events(root):
    return Counter(e["event"] for e in read_events(root))


def render(x):
    return MarkerCamera(lambda: [x], [(0, 20)]).capture()


class Scripted:
    """Plays a fixed list of action indices, then repeats the last."""

    def __init__(self, indices):
        self.indices = list(indices)

    def choose(self, state, rng):
        return self.indices.pop(0) if len(self.indices) > 1 else self.indices[0]

    def observe(self, state, action_index, result):
        pass


class FlakyCamera:
    """Wraps a camera; raises on the capture numbers listed in ``fail_on`` (1-based)."""

    def __init__(self, camera, fail_on, error=RuntimeError):
        self.camera, self.fail_on, self.error, self.calls = camera, set(fail_on), error, 0

    @property
    def resolution(self):
        return self.camera.resolution

    def capture(self):
        self.calls += 1
        if self.calls in self.fail_on:
            raise self.error(f"capture {self.calls} failed")
        return self.camera.capture()


class LostReplyTransport(FakeTransport):
    """The device executes command number ``lose`` but its reply only arrives after the timeout."""

    def __init__(self, device, lose):
        super().__init__(self._respond)
        self.device, self.lose, self.count, self.late = device, set(lose), 0, None

    def _respond(self, line):
        self.count += 1
        reply = self.device(line)
        if self.count in self.lose:
            self.late = reply
            return None
        return reply

    def receive(self, timeout=None):
        try:
            return super().receive(timeout)
        except TimeoutError:
            if self.late is not None:
                self.queue_reply(self.late)  # turns up after the caller gave up
                self.late = None
            raise


class Crash(BaseException):
    pass


def test_collects_the_target_and_passes_verify(tmp_path, marker_rig_factory):
    summary = run(marker_rig_factory(), tmp_path, num_samples=50, steps_per_episode=10)
    assert summary.total == summary.samples == 50
    assert summary.episodes == 5
    report = verify(tmp_path)
    assert report.ok, report.format()
    assert report.samples == 50 and report.episodes == 5


def test_an_episode_of_n_steps_stores_n_plus_one_frames(tmp_path, marker_rig_factory):
    run(marker_rig_factory(), tmp_path, num_samples=30, steps_per_episode=10)
    for episode in (tmp_path / "frames").iterdir():
        assert sorted(p.name for p in episode.iterdir()) == [f"{i:03d}.png" for i in range(11)]


def test_states_are_what_the_rig_reported_and_photos_show_them(tmp_path, marker_rig_factory):
    rig = marker_rig_factory()
    run(rig, tmp_path, num_samples=40, steps_per_episode=8)
    samples = read_samples(tmp_path)
    assert samples[-1].state_after == {"x": rig.transport.responder.x}
    for s in samples:
        assert torch.equal(load_frame(tmp_path / s.before_frame), render(s.state_before["x"]))
        assert torch.equal(load_frame(tmp_path / s.after_frame), render(s.state_after["x"]))
        assert s.command == f"MOVE {int(s.action[0])}"
        assert s.state_after_tensor == [float(s.state_after["x"])]


def test_photos_differ_exactly_when_the_state_changed(tmp_path, marker_rig_factory):
    policy = EdgeAwareRandomWalk(marker_rig_factory().action_space.actions(), reject_rate=0.3)
    run(marker_rig_factory(), tmp_path, num_samples=80, steps_per_episode=20, policy=policy)
    samples = read_samples(tmp_path)
    assert any(s.changed for s in samples) and any(not s.changed for s in samples)
    for s in samples:
        same = torch.equal(load_frame(tmp_path / s.before_frame), load_frame(tmp_path / s.after_frame))
        assert same == (not s.changed)
        assert s.changed == (s.state_after != s.state_before)


def test_out_of_range_move_is_recorded_as_rejected(tmp_path, marker_rig_factory):
    # keep moving +5 until the rail ends; the rig answers "ERR out of range"
    summary = run(marker_rig_factory(), tmp_path, num_samples=6, steps_per_episode=6, policy=Scripted([3]))
    samples = read_samples(tmp_path)
    statuses = [s.status for s in samples]
    first_rejected = statuses.index("rejected")
    assert set(statuses[first_rejected:]) == {"rejected"} and set(statuses[:first_rejected]) <= {"ok"}
    for s in samples[first_rejected:]:
        assert s.state_after == s.state_before and s.state_before["x"] > 15
        assert not s.changed and s.error == "out of range"
    assert summary.rejected == events(tmp_path)["rejected"] == len(samples) - first_rejected
    assert verify(tmp_path).ok


def test_lost_reply_ends_the_episode_and_resyncs_without_resending(tmp_path, marker_rig_factory):
    rig = marker_rig_factory()
    device = rig.transport.responder
    rig.transport = LostReplyTransport(device, lose={4})  # sync move is command 1
    summary = run(rig, tmp_path, num_samples=20, steps_per_episode=10)

    assert summary.total == 20
    assert summary.failed_moves == 1
    seen = events(tmp_path)
    assert seen["move_failed"] == 1
    assert seen["stale_replies_drained"] == 1
    assert seen["synced"] == 2
    samples = read_samples(tmp_path)
    assert [s.step_index for s in samples if s.episode_id == "ep000000"] == [0, 1]
    assert samples[-1].state_after == {"x": device.x}
    assert len(rig.transport.sent) == 2 + 20 + 1  # two syncs, the samples, and the lost move, never resent
    assert verify(tmp_path).ok


def test_rig_that_never_answers_stops_the_run(tmp_path, marker_rig_factory):
    rig = marker_rig_factory()
    rig.transport = FakeTransport(lambda line: None)
    summary = run(rig, tmp_path, num_samples=20, sync_attempts=2, max_failed_episodes=3)
    assert summary.total == 0
    assert summary.stopped == "3 episodes in a row failed"
    assert events(tmp_path)["sync_gave_up"] == 3


def test_transient_camera_failure_is_retried(tmp_path, marker_rig_factory):
    rig = marker_rig_factory()
    camera = FlakyCamera(rig.camera, fail_on={3})
    summary = run(rig, tmp_path, num_samples=10, steps_per_episode=10, camera=camera)
    assert summary.total == 10 and summary.camera_failures == 1
    assert summary.episodes == 1
    assert verify(tmp_path).ok


def test_persistent_camera_failure_ends_the_episode_without_a_half_written_sample(tmp_path, marker_rig_factory):
    rig = marker_rig_factory()
    camera = FlakyCamera(rig.camera, fail_on={4, 5, 6})  # frame 3 of episode 0, all three tries
    summary = run(rig, tmp_path, num_samples=10, steps_per_episode=10, camera=camera)
    samples = read_samples(tmp_path)
    assert [s.step_index for s in samples if s.episode_id == "ep000000"] == [0, 1]
    assert not (tmp_path / "frames/ep000000/003.png").exists()
    assert summary.total == 10 and summary.episodes == 2
    assert events(tmp_path)["episode_ended"] == 1
    assert verify(tmp_path).ok


def test_crash_then_resume_has_no_duplicates_or_gaps(tmp_path, marker_rig_factory):
    rig = marker_rig_factory()
    camera = FlakyCamera(rig.camera, fail_on={30}, error=Crash)
    with pytest.raises(Crash):
        run(rig, tmp_path, num_samples=100, steps_per_episode=10, camera=camera)
    crashed_at = len(read_samples(tmp_path))
    assert 0 < crashed_at < 100
    with open(tmp_path / MANIFEST, "ab") as f:
        f.write(b'{"sample_id":"ep0000')  # the write that was cut off

    summary = run(marker_rig_factory(), tmp_path, num_samples=100, steps_per_episode=10, resume=True)

    assert summary.existing == crashed_at and summary.total == 100
    samples = read_samples(tmp_path)
    assert len(samples) == 100
    assert len({s.sample_id for s in samples}) == 100
    seen = events(tmp_path)
    assert seen["truncated_line_dropped"] == 1 and seen["resumed"] == 1
    report = verify(tmp_path)
    assert report.ok and not report.warnings, report.format()
    # the interrupted episode stays as a shorter episode; the resumed run starts a new one
    by_episode = Counter(s.episode_id for s in samples)
    assert by_episode["ep000002"] == crashed_at - 20
    assert "ep000003" in by_episode
    for episode, n in by_episode.items():
        assert [s.step_index for s in samples if s.episode_id == episode] == list(range(n))


def test_resume_counts_toward_the_target_total(tmp_path, marker_rig_factory):
    run(marker_rig_factory(), tmp_path, num_samples=40, steps_per_episode=10)
    summary = run(marker_rig_factory(), tmp_path, num_samples=100, steps_per_episode=10, resume=True)
    assert summary.existing == 40 and summary.samples == 60
    assert len(read_samples(tmp_path)) == 100
    episodes = sorted({s.episode_id for s in read_samples(tmp_path)})
    assert episodes == [f"ep{i:06d}" for i in range(10)]


def test_existing_dataset_needs_resume(tmp_path, marker_rig_factory):
    run(marker_rig_factory(), tmp_path, num_samples=5)
    with pytest.raises(FileExistsError, match="resume"):
        run(marker_rig_factory(), tmp_path, num_samples=10)


def test_resume_refuses_a_different_camera(tmp_path, marker_rig_factory):
    run(marker_rig_factory(), tmp_path, num_samples=5)
    rig = marker_rig_factory()
    rig.camera = MarkerCamera(lambda: [0], [(0, 20)], resolution=(64, 64))
    with pytest.raises(DatasetFormatError, match="resolution"):
        run(rig, tmp_path, num_samples=10, resume=True)


def test_stop_request_finishes_the_current_step(tmp_path, marker_rig_factory):
    rig = marker_rig_factory()
    with DatasetWriter.create(tmp_path, build_meta(rig, rig.camera)) as writer:
        policy = EdgeAwareRandomWalk(rig.action_space.actions())
        collector = Collector(rig, policy, writer, CollectorConfig(settle_time=0))
        collector.on_progress = lambda s: s.samples == 7 and collector.request_stop("test")
        summary = collector.run(handle_signals=False)
    assert summary.samples == 7 and summary.stopped == "test"
    assert not rig.transport.is_open
    assert verify(tmp_path).ok


def test_settle_time_is_waited_before_every_photo(tmp_path, marker_rig_factory):
    rig = marker_rig_factory()
    waits = []
    with DatasetWriter.create(tmp_path, build_meta(rig, rig.camera)) as writer:
        config = CollectorConfig(num_samples=5, steps_per_episode=5, settle_time=0.25)
        Collector(rig, EdgeAwareRandomWalk(rig.action_space.actions()), writer, config, sleep=waits.append).run(
            handle_signals=False
        )
    assert waits == [0.25] * 6


def test_summary_reports_the_no_change_fraction(tmp_path, marker_rig_factory):
    summary = run(marker_rig_factory(), tmp_path, num_samples=8, steps_per_episode=8, policy=Scripted([3]))
    unchanged = sum(not s.changed for s in read_samples(tmp_path))
    assert unchanged > 0
    assert summary.no_change_fraction == unchanged / 8
    assert f"no change {unchanged / 8:.0%}" in summary.format()


def test_acceptance_300_samples_verify_and_export(tmp_path, marker_rig_factory):
    root, out = tmp_path / "dataset", tmp_path / "train"
    summary = run(marker_rig_factory(), root, num_samples=300, steps_per_episode=20)
    assert summary.total == 300
    assert verify(root).ok

    splits = ensure_splits(root)
    assert set(splits.values()) == {"train", "val", "test"}
    exported = export(root, out, split="train")
    assert exported == sum(1 for s in read_samples(root) if splits[s.episode_id] == "train")
    report = verify(out)
    assert report.ok and report.samples == exported, report.format()
    assert {s.episode_id for s in read_samples(out)} == {e for e, name in splits.items() if name == "train"}


def test_collector_has_no_rig_specific_code():
    data = Path(__file__).parents[2] / "src" / "vjepa_byor" / "data"
    for path in data.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert not re.search(r"esp32|stepper", text), path.name
