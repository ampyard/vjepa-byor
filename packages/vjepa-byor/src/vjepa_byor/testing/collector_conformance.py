"""Checks that the data collector can drive a rig plugin end to end, without hardware.

Mix :class:`CollectorConformanceTests` into the same test class as
:class:`~vjepa_byor.testing.conformance.RigConformanceTests`; it uses the same
``rig`` fixture, which needs a camera (see
:class:`~vjepa_byor.testing.fake_camera.MarkerCamera`)::

    class TestMyRig(RigConformanceTests, CollectorConformanceTests):
        ...
"""

from __future__ import annotations

import pytest
import torch

from vjepa_byor.data import CollectorConfig, CollectRun, collect, load_frame, read_samples, verify
from vjepa_byor.rig import Rig


class CollectorConformanceTests:
    collect_samples: int = 24
    collect_steps_per_episode: int = 6

    @pytest.fixture
    def rig(self) -> Rig:
        pytest.fail("override the `rig` fixture to return the Rig under test")

    def collect_dataset(self, rig: Rig, root):
        """Run a short collection into ``root``; skips if the rig has no camera."""
        if rig.camera is None:
            pytest.skip("rig has no camera")
        config = CollectorConfig(
            num_samples=self.collect_samples, steps_per_episode=self.collect_steps_per_episode, settle_time=0
        )
        return collect(rig, root, CollectRun(config=config, handle_signals=False))

    @pytest.fixture
    def collected(self, rig: Rig, tmp_path):
        return tmp_path / "dataset", self.collect_dataset(rig, tmp_path / "dataset")

    def test_collects_the_requested_samples(self, collected) -> None:
        root, summary = collected
        assert summary.total == self.collect_samples, summary.format()
        assert len(read_samples(root)) == self.collect_samples

    def test_collected_dataset_verifies(self, collected) -> None:
        report = verify(collected[0])
        assert report.ok, report.format()

    def test_recorded_states_use_the_rig_state_type(self, rig: Rig, collected) -> None:
        for s in read_samples(collected[0]):
            for data, tensor in ((s.state_before, s.state_before_tensor), (s.state_after, s.state_after_tensor)):
                state = rig.state_type.from_dict(data)
                assert state.as_tensor().tolist() == tensor

    def test_frames_have_the_camera_resolution(self, rig: Rig, collected) -> None:
        root = collected[0]
        first = read_samples(root)[0]
        assert load_frame(root / first.before_frame).shape == (3, *rig.camera.resolution)
        assert load_frame(root / first.before_frame).dtype == torch.uint8
