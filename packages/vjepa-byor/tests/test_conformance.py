import pytest
import torch

from vjepa_byor import interfaces, register_rig, unregister_rig
from vjepa_byor.testing.conformance import RigConformanceTests


class TestLineRig(RigConformanceTests):
    registered_name = "line"

    @pytest.fixture(autouse=True)
    def registered(self, line_rig_factory):
        register_rig("line", line_rig_factory)
        yield
        unregister_rig("line")

    @pytest.fixture
    def rig(self, line_rig):
        return line_rig

    @pytest.fixture
    def sample_states(self, line_states):
        return line_states


class TestDirectLineRig(RigConformanceTests):
    """The same toy rig driven through an ActionTransport: the codec tests are skipped."""

    registered_name = "line-direct"

    @pytest.fixture(autouse=True)
    def registered(self, direct_rig_factory):
        register_rig("line-direct", direct_rig_factory)
        yield
        unregister_rig("line-direct")

    @pytest.fixture
    def rig(self, direct_rig):
        return direct_rig

    @pytest.fixture
    def sample_states(self, line_states):
        return line_states


def test_direct_rig_satisfies_the_protocols(direct_rig):
    assert direct_rig.codec is None
    assert isinstance(direct_rig.transport, interfaces.ActionTransport)
    assert not isinstance(direct_rig.transport, interfaces.Transport)


def test_toy_rig_satisfies_the_protocols(line_rig, line_states):
    assert isinstance(line_rig.action_space, interfaces.ActionSpace)
    assert isinstance(line_rig.mapping, interfaces.VJepaMapping)
    assert isinstance(line_rig.codec, interfaces.CommandCodec)
    assert isinstance(line_rig.transport, interfaces.Transport)
    assert isinstance(line_rig.camera, interfaces.Camera)
    assert isinstance(line_states[0], interfaces.RigState)


def test_suite_catches_a_lossy_codec(line_rig):
    line_rig.codec.encode_action = lambda action: "MOVE 0"
    with pytest.raises(AssertionError):
        RigConformanceTests().test_codec_action_round_trip(line_rig)


def test_suite_catches_a_mapping_that_merges_actions(line_rig):
    line_rig.mapping.map_actions = lambda actions: torch.zeros((*actions.shape[:-1], 7))
    with pytest.raises(AssertionError):
        RigConformanceTests().test_mapping_keeps_actions_distinct(line_rig)
