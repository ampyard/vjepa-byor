"""Error paths and less common branches of the data package, the CLI and the testing helpers."""

import json
import random
import signal
import subprocess
import sys
import threading
import types

import pytest
import torch

from vjepa_byor import cli, register_rig, unregister_rig
from vjepa_byor.data import (
    CollectedDataset,
    CollectionSummary,
    Collector,
    CollectorConfig,
    CollectRun,
    DatasetFormatError,
    DatasetWriter,
    EdgeAwareRandomWalk,
    build_meta,
    collect,
    ensure_splits,
    episodes,
    read_samples,
    verify,
)
from vjepa_byor.data import collector as collector_module
from vjepa_byor.data.storage import MANIFEST, encode_png, read_events, read_meta
from vjepa_byor.testing import FakeTransport
from vjepa_byor.testing.fake_camera import MarkerCamera


def run(rig, root, *, resume=False, rig_options=None, **config):
    config = CollectorConfig(**{"settle_time": 0, **config})
    return collect(
        rig, root, CollectRun(config=config, resume=resume, rig_options=rig_options, handle_signals=False)
    )


def events(root):
    return [e["event"] for e in read_events(root)]


DEVICE = object()


def scripted_transport(rig, replies):
    """Replace the rig's transport: the first replies come from ``replies``, then the device.

    ``None`` means no reply (a timeout) and ``DEVICE`` lets the device answer.
    """
    device = rig.transport.responder
    queue = list(replies)

    def respond(line):
        reply = queue.pop(0) if queue else DEVICE
        return device(line) if reply is DEVICE else reply

    rig.transport = FakeTransport(respond)
    return device


def new_collector(rig, root, config=None, **kwargs):
    writer = DatasetWriter.create(root, build_meta(rig, rig.camera))
    policy = EdgeAwareRandomWalk(rig.action_space.actions())
    return writer, Collector(rig, policy, writer, config or CollectorConfig(settle_time=0), **kwargs)


# Collector


def test_summary_before_the_first_sample():
    summary = CollectionSummary(target=10, existing=0)
    assert summary.eta is None and summary.no_change_fraction == 0.0
    assert "eta ?" in summary.format()


def test_a_camera_is_required(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    rig.camera = None
    with pytest.raises(ValueError, match="no camera"):
        collect(rig, tmp_path)
    writer = DatasetWriter.create(tmp_path / "w", build_meta(marker_rig_factory(), marker_rig_factory().camera))
    with writer, pytest.raises(ValueError, match="no camera"):
        Collector(rig, EdgeAwareRandomWalk(rig.action_space.actions()), writer)


def test_camera_passed_explicitly_replaces_the_rigs(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    camera = rig.camera
    rig.camera = None
    config = CollectorConfig(num_samples=5, settle_time=0)
    summary = collect(rig, tmp_path, CollectRun(config=config, camera=camera, handle_signals=False))
    assert summary.total == 5


def test_reply_timeout_overrides_the_rigs(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    writer, _ = new_collector(rig, tmp_path, CollectorConfig(reply_timeout=1.5))
    writer.close()
    assert rig.reply_timeout == 1.5


def test_a_rig_opened_by_the_caller_stays_open(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    with rig:
        run(rig, tmp_path, num_samples=5)
        assert rig.transport.is_open


def test_camera_failing_on_the_first_frame_ends_the_episode(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    calls = {"n": 0}
    real = rig.camera.capture

    def capture():
        calls["n"] += 1
        if calls["n"] <= 3:  # frame 0 of episode 0, every try
            raise RuntimeError("lens cap on")
        return real()

    rig.camera.capture = capture
    summary = run(rig, tmp_path, num_samples=5, steps_per_episode=5)
    assert summary.total == 5 and summary.episodes == 2 and summary.camera_failures == 3
    assert {s.episode_id for s in read_samples(tmp_path)} == {"ep000001"}


def test_camera_returning_the_wrong_shape_counts_as_a_failure(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    rig.camera.capture = lambda: torch.zeros((3, 8, 8), dtype=torch.uint8)
    summary = run(rig, tmp_path, num_samples=5, max_failed_episodes=2)
    assert summary.total == 0 and summary.camera_failures == 6
    errors = [e["error"] for e in read_events(tmp_path) if e["event"] == "camera_failed"]
    assert "expected uint8 (3, 32, 48)" in errors[0]


def test_sync_tries_other_moves_after_a_rejection(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    scripted_transport(rig, ["ERR busy", "ERR busy"])
    summary = run(rig, tmp_path, num_samples=5)
    assert summary.total == 5
    assert events(tmp_path).count("sync_rejected") == 2
    sync_commands = rig.transport.sent[:3]
    assert len(set(sync_commands)) == 3  # a different action each try


def test_sync_survives_a_timeout(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    scripted_transport(rig, [None])
    summary = run(rig, tmp_path, num_samples=5)
    assert summary.total == 5 and summary.failed_moves == 1
    assert "sync_failed" in events(tmp_path)


def test_garbled_reply_is_a_failed_move(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    device = scripted_transport(rig, [DEVICE, "\x00garbage"])  # sync answers, step 0 does not parse
    summary = run(rig, tmp_path, num_samples=5)
    assert summary.failed_moves == 1 and summary.total == 5
    failure = next(e for e in read_events(tmp_path) if e["event"] == "move_failed")
    assert "CodecError" in failure["error"]


def test_drain_stops_after_a_bounded_number_of_stale_replies(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    writer, collector = new_collector(rig, tmp_path)
    with writer:
        rig.open()
        for _ in range(150):
            rig.transport.queue_reply("POS 1")
        collector._drain()
    drained = next(e for e in read_events(tmp_path) if e["event"] == "stale_replies_drained")
    assert len(drained["lines"]) == 100


def test_first_signal_stops_cleanly_and_a_second_ctrl_c_aborts(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    writer, collector = new_collector(rig, tmp_path / "a")
    collector.on_progress = lambda s: s.samples == 4 and signal.raise_signal(signal.SIGTERM)
    with writer:
        summary = collector.run()
    assert summary.samples == 4 and summary.stopped == "received SIGTERM"
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # handlers restored

    rig = marker_rig_factory()
    writer, collector = new_collector(rig, tmp_path / "b")

    def interrupt_twice(s):
        if s.samples == 3:
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)

    collector.on_progress = interrupt_twice
    with writer, pytest.raises(KeyboardInterrupt):
        collector.run()
    assert not rig.transport.is_open


def test_signal_handling_is_skipped_off_the_main_thread(marker_rig_factory, tmp_path):
    rig = marker_rig_factory()
    writer, collector = new_collector(rig, tmp_path, CollectorConfig(num_samples=5, settle_time=0))
    results, errors = [], []

    def target():
        try:
            results.append(collector.run())
        except BaseException as e:
            errors.append(e)

    with writer:
        thread = threading.Thread(target=target)
        thread.start()
        thread.join()
    assert not errors and results[0].total == 5


def test_collector_version_when_the_package_metadata_is_missing(monkeypatch):
    def missing(name):
        raise collector_module.PackageNotFoundError(name)

    monkeypatch.setattr(collector_module, "version", missing)
    assert collector_module._collector_version() == "unknown"


def test_resume_with_different_rig_options_warns(marker_rig_factory, tmp_path):
    run(marker_rig_factory(), tmp_path, num_samples=5, rig_options={"port": "COM3"})
    run(marker_rig_factory(), tmp_path, num_samples=10, resume=True, rig_options={"port": "COM4"})
    resumed = next(e for e in read_events(tmp_path) if e["event"] == "resumed")
    assert "rig config differs" in resumed["warning"]
    assert len(read_samples(tmp_path)) == 10


def test_resume_refuses_another_rig(marker_rig_factory, tmp_path):
    run(marker_rig_factory(), tmp_path, num_samples=5)
    meta = read_meta(tmp_path)
    other = type(meta).from_dict({**meta.to_dict(), "rig": "other"})
    with pytest.raises(DatasetFormatError, match="rig is 'other'"):
        DatasetWriter.resume(tmp_path, other)


# Policy


def test_policy_falls_back_to_any_action_when_all_are_known_rejected(line_rig):
    state = line_rig.state_type(10)
    policy = EdgeAwareRandomWalk(line_rig.action_space.actions(), reject_rate=0)
    for index in range(4):
        policy.observe(state, index, None)
    assert policy.choose(state, random.Random(0)) in range(4)


# Storage and records


def test_encode_png_refuses_a_non_image():
    with pytest.raises(ValueError, match="uint8"):
        encode_png(torch.zeros((3, 4, 4)))
    with pytest.raises(ValueError):
        encode_png(torch.zeros((4, 4), dtype=torch.uint8))


def test_meta_that_is_not_json(tmp_path):
    (tmp_path / "meta.json").write_text("{nope")
    with pytest.raises(DatasetFormatError, match="not valid JSON"):
        read_meta(tmp_path)


def test_records_ignore_unknown_fields(marker_rig_factory):
    rig = marker_rig_factory()
    meta = build_meta(rig, rig.camera)
    assert type(meta).from_dict({**meta.to_dict(), "added_in_a_later_version": 1}) == meta


def test_events_of_a_missing_dataset_are_empty(tmp_path):
    assert read_events(tmp_path) == []


# Dataset


@pytest.fixture
def small(tmp_path, marker_rig_factory):
    run(marker_rig_factory(), tmp_path / "ds", num_samples=20, steps_per_episode=10)
    return tmp_path / "ds"


def test_two_episodes_all_go_to_train(small):
    assert set(ensure_splits(small).values()) == {"train"}


def test_bad_ratios_and_split_names(small):
    with pytest.raises(ValueError, match="ratios"):
        ensure_splits(small, ratios=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="split must be one of"):
        CollectedDataset(small, "holdout")


def test_episode_summaries(small):
    infos = episodes(small)
    samples = read_samples(small)
    assert [i.episode_id for i in infos] == ["ep000000", "ep000001"]
    assert sum(i.num_samples for i in infos) == 20
    assert sum(i.num_changed for i in infos) == sum(s.changed for s in samples)
    assert sum(i.num_rejected for i in infos) == sum(s.status == "rejected" for s in samples)


def rewrite(root, edit):
    records = [json.loads(line) for line in (root / MANIFEST).read_text().splitlines()]
    edit(records)
    (root / MANIFEST).write_text("".join(json.dumps(r) + "\n" for r in records))


def test_verify_flags_unknown_status_and_wrong_resolution(small):
    def edit(records):
        records[0]["status"] = "failed"
        records[1]["resolution"] = [1, 1]

    rewrite(small, edit)
    errors = verify(small).errors
    assert any("unknown status 'failed'" in e for e in errors)
    assert any("resolution [1, 1] differs" in e for e in errors)


def test_verify_reports_a_corrupt_manifest(small):
    (small / MANIFEST).write_text("garbage\n" + (small / MANIFEST).read_text())
    report = verify(small)
    assert not report.ok and "corrupt line" in report.errors[0]


def test_verify_report_caps_the_listed_errors(tmp_path, marker_rig_factory):
    run(marker_rig_factory(), tmp_path, num_samples=40)
    for path in (tmp_path / "frames").rglob("*.png"):
        path.unlink()
    report = verify(tmp_path)
    assert len(report.errors) == 80  # two missing frames per sample
    text = report.format()
    assert text.startswith("FAILED") and "... and 30 more errors" in text


# MarkerCamera


def test_marker_camera_draws_one_or_two_coordinates():
    x = {"v": 0.0}
    one_d = MarkerCamera(lambda: [x["v"]], [(0, 10)], resolution=(9, 13), marker=3)
    image = one_d.capture()
    assert image.shape == (3, 9, 13) and image.dtype == torch.uint8
    assert image[0].nonzero()[0].tolist() == [3, 0]  # vertically centred, at the left edge
    x["v"] = 10
    assert one_d.capture()[0].nonzero()[-1].tolist() == [5, 12]  # right edge

    two_d = MarkerCamera(lambda: [10, 0], [(0, 10), (0, 10)], resolution=(9, 13), marker=3)
    assert two_d.capture()[0].nonzero()[0].tolist() == [0, 10]


def test_marker_camera_clamps_positions_outside_the_bounds():
    camera = MarkerCamera(lambda: [-50], [(0, 10)], resolution=(9, 13))
    assert camera.capture()[0].nonzero()[0].tolist()[1] == 0


def test_marker_camera_refuses_three_coordinates():
    with pytest.raises(ValueError):
        MarkerCamera(lambda: [0, 0, 0], [(0, 1)] * 3)


# CLI


@pytest.fixture
def line_registered(marker_rig_factory):
    register_rig("line", marker_rig_factory)
    yield
    unregister_rig("line")


def test_cli_rig_config_from_yaml(tmp_path, monkeypatch, line_registered):
    (tmp_path / "rig.yaml").write_text("start: 0\n")
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=lambda text: {"start": 0}))
    argv = ["collect", "--rig", "line", "--out", str(tmp_path / "ds"), "--settle-time", "0", "--num-samples", "3"]
    assert cli.main([*argv, "--rig-config", str(tmp_path / "rig.yaml")]) == 0


def test_cli_yaml_without_pyyaml(tmp_path, monkeypatch):
    (tmp_path / "rig.yml").write_text("start: 0\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(SystemExit, match="needs PyYAML"):
        cli.main(["collect", "--rig", "line", "--out", str(tmp_path / "ds"), "--rig-config", str(tmp_path / "rig.yml")])


def test_cli_rig_config_must_be_a_mapping(tmp_path):
    (tmp_path / "rig.json").write_text("[1, 2]")
    with pytest.raises(SystemExit, match="mapping"):
        cli.main(["collect", "--rig", "line", "--out", str(tmp_path / "ds"), "--rig-config", str(tmp_path / "rig.json")])


def test_cli_progress_on_a_terminal(tmp_path, capsys, monkeypatch, line_registered):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    argv = ["collect", "--rig", "line", "--out", str(tmp_path / "ds"), "--settle-time", "0", "--num-samples", "3"]
    assert cli.main(argv) == 0
    assert "\r1/3 samples" in capsys.readouterr().out


def test_cli_progress_when_piped(tmp_path, capsys, line_registered):
    argv = ["collect", "--rig", "line", "--out", str(tmp_path / "ds"), "--settle-time", "0", "--num-samples", "100"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "50/100 samples" in out and "\r" not in out


def test_cli_ctrl_c_prints_what_was_collected(tmp_path, capsys, monkeypatch, line_registered):
    def interrupted(rig, out, run):
        run.on_progress(CollectionSummary(target=10, existing=0, samples=4))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "collect", interrupted)
    assert cli.main(["collect", "--rig", "line", "--out", str(tmp_path / "ds")]) == 130
    out = capsys.readouterr().out
    assert "aborted" in out and "4/10 samples" in out


def test_cli_reports_an_early_stop(tmp_path, capsys, monkeypatch, line_registered):
    monkeypatch.setattr(cli, "collect", lambda rig, out, run: CollectionSummary(10, 0, stopped="received SIGTERM"))
    assert cli.main(["collect", "--rig", "line", "--out", str(tmp_path / "ds")]) == 0
    assert "stopped early: received SIGTERM" in capsys.readouterr().out


def test_cli_runs_as_a_module(tmp_path):
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "vjepa_byor.cli", "collect", "--verify", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "meta.json is missing" in result.stdout
