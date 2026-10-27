"""The fake-sim rig end to end, until the real sim rig (module 11) can take over."""

import pytest
import torch

from vjepa_byor import interfaces, register_rig, unregister_rig
from vjepa_byor.cli import main
from vjepa_byor.data import CollectedDataset, ensure_splits, export, load_frame, read_samples, verify
from vjepa_byor.testing.collector_conformance import CollectorConformanceTests
from vjepa_byor.testing.conformance import RigConformanceTests

BOUNDS = "0:60,0:40"


@pytest.fixture
def registered(fake_sim_factory):
    register_rig("fake-sim", fake_sim_factory)
    yield
    unregister_rig("fake-sim")


class TestFakeSim(RigConformanceTests, CollectorConformanceTests):
    registered_name = "fake-sim"

    @pytest.fixture(autouse=True)
    def _registered(self, registered):
        pass

    @pytest.fixture
    def rig(self, fake_sim):
        return fake_sim

    @pytest.fixture
    def sample_states(self, fake_sim):
        state = fake_sim.state_type
        return [state(0, 0), state(30, 20), state(60, 40)]


class TestMarkerLineRig(CollectorConformanceTests):
    """The collector conformance suite also passes for the 1-D toy rig."""

    @pytest.fixture
    def rig(self, marker_rig_factory):
        return marker_rig_factory()


def test_collector_conformance_skips_a_rig_without_a_camera(line_rig, tmp_path):
    line_rig.camera = None
    with pytest.raises(pytest.skip.Exception):
        CollectorConformanceTests().collect_dataset(line_rig, tmp_path / "ds")


def test_fake_sim_satisfies_the_protocols(fake_sim):
    assert isinstance(fake_sim.action_space, interfaces.ActionSpace)
    assert isinstance(fake_sim.mapping, interfaces.VJepaMapping)
    assert isinstance(fake_sim.codec, interfaces.CommandCodec)
    assert isinstance(fake_sim.camera, interfaces.Camera)
    assert len(fake_sim.action_space) == 24


def test_fake_sim_camera_gives_every_position_its_own_photo(fake_sim):
    device = fake_sim.transport.responder
    seen = set()
    for x in range(0, 61):
        for y in range(0, 41):
            device.x, device.y = x, y
            seen.add(tuple(fake_sim.camera.capture()[0].nonzero()[0].tolist()))  # marker's top-left pixel
    assert len(seen) == 61 * 41


@pytest.fixture
def cli_dataset(tmp_path_factory, fake_sim_factory):
    """300 samples collected through the CLI, the way module 11's test will do it."""
    register_rig("fake-sim", fake_sim_factory)
    try:
        out = tmp_path_factory.mktemp("cli") / "fake-sim"
        argv = ["collect", "--rig", "fake-sim", "--out", str(out), "--num-samples", "300"]
        assert main([*argv, "--settle-time", "0", "--state-bounds", BOUNDS]) == 0
    finally:
        unregister_rig("fake-sim")
    return out


def test_cli_collects_300_samples_that_verify(cli_dataset):
    report = verify(cli_dataset)
    assert report.ok and report.samples == 300 and report.episodes == 15, report.format()


def test_cli_dataset_photos_match_the_reported_positions(cli_dataset, fake_sim_factory):
    rig = fake_sim_factory()
    device = rig.transport.responder
    for s in read_samples(cli_dataset)[:60]:
        device.x, device.y = s.state_after["x"], s.state_after["y"]
        assert torch.equal(load_frame(cli_dataset / s.after_frame), rig.camera.capture())
        same = torch.equal(load_frame(cli_dataset / s.before_frame), load_frame(cli_dataset / s.after_frame))
        assert same == (not s.changed)


def test_cli_dataset_stays_in_bounds_and_uses_the_whole_vocabulary(cli_dataset):
    samples = read_samples(cli_dataset)
    for s in samples:
        assert 0 <= s.state_after["x"] <= 60 and 0 <= s.state_after["y"] <= 40
    assert len({s.action_index for s in samples}) == 24
    rejected = sum(s.status == "rejected" for s in samples)
    assert rejected < 0.1 * len(samples)  # with bounds, only the deliberate 2% and the first tries


def test_cli_dataset_splits_and_exports(cli_dataset, tmp_path):
    splits = ensure_splits(cli_dataset)
    assert set(splits.values()) == {"train", "val", "test"}
    n = export(cli_dataset, tmp_path / "train", split="train")
    assert n == len(CollectedDataset(cli_dataset, "train"))
    assert verify(tmp_path / "train").ok


def test_cli_resume_with_fake_sim(tmp_path, registered):
    out = str(tmp_path / "ds")
    base = ["collect", "--rig", "fake-sim", "--out", out, "--settle-time", "0"]
    assert main([*base, "--num-samples", "40"]) == 0
    assert main([*base, "--num-samples", "90", "--resume"]) == 0
    assert len(read_samples(out)) == 90
    assert verify(out).ok
