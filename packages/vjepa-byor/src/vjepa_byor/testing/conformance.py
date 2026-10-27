"""Conformance tests every rig plugin should pass.

Subclass :class:`RigConformanceTests` in your plugin's test suite and override
the ``rig`` and ``sample_states`` fixtures. Back the rig with a
:class:`~vjepa_byor.testing.FakeTransport` whose responder answers like your
device, so the suite runs without hardware::

    from vjepa_byor.testing import FakeTransport
    from vjepa_byor.testing.conformance import RigConformanceTests

    class TestMyRig(RigConformanceTests):
        registered_name = "my-rig"  # also checks the entry point; omit to skip

        @pytest.fixture
        def rig(self):
            return make_rig(transport=FakeTransport(MyDeviceEmulator()))

        @pytest.fixture
        def sample_states(self):
            return [MyState(0, 0), MyState(120, -40)]

``test_execute_every_action`` sends every action once, in order, from the
emulator's starting state, so start it somewhere all of them are in range.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar

import pytest
import torch

from vjepa_byor.interfaces import CodecError, RigState
from vjepa_byor.registry import available_rigs, load_rig
from vjepa_byor.rig import Rig

_RIG_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class RigConformanceTests:
    registered_name: ClassVar[str | None] = None

    @pytest.fixture
    def rig(self) -> Rig:
        pytest.fail("override the `rig` fixture to return the Rig under test")

    @pytest.fixture
    def sample_states(self) -> list[RigState]:
        pytest.fail("override the `sample_states` fixture to return a few example states")

    @pytest.fixture
    def open_rig(self, rig: Rig):
        with rig:
            yield rig

    # Rig bundle

    def test_name_is_a_slug(self, rig: Rig) -> None:
        assert _RIG_NAME.match(rig.name), f"rig name {rig.name!r} should be lowercase-with-dashes"

    def test_registered_under_its_name(self) -> None:
        if self.registered_name is None:
            pytest.skip("registered_name not set")
        assert self.registered_name in available_rigs()
        assert load_rig(self.registered_name).name == self.registered_name

    # ActionSpace

    def test_actions_shape(self, rig: Rig) -> None:
        space = rig.action_space
        actions = space.actions()
        assert len(space) > 0
        assert actions.is_floating_point()
        assert actions.shape == (len(space), space.dim)

    def test_actions_are_unique(self, rig: Rig) -> None:
        actions = rig.action_space.actions()
        assert len(torch.unique(actions, dim=0)) == len(actions)

    def test_labels_match_actions(self, rig: Rig) -> None:
        labels = rig.action_space.labels()
        assert len(labels) == len(rig.action_space)
        assert len(set(labels)) == len(labels)

    def test_contains_every_action(self, rig: Rig) -> None:
        for action in rig.action_space.actions():
            assert rig.action_space.contains(action)

    def test_rejects_foreign_actions(self, rig: Rig) -> None:
        dim = rig.action_space.dim
        assert not rig.action_space.contains(torch.full((dim,), float("nan")))
        assert not rig.action_space.contains(torch.zeros(dim + 1))

    # RigState

    def test_sample_states_have_the_rig_state_type(self, rig: Rig, sample_states: list[RigState]) -> None:
        assert sample_states
        for state in sample_states:
            assert isinstance(state, rig.state_type)

    def test_state_tensor_is_1d_float(self, sample_states: list[RigState]) -> None:
        sizes = set()
        for state in sample_states:
            tensor = state.as_tensor()
            assert tensor.ndim == 1 and tensor.is_floating_point()
            sizes.add(len(tensor))
        assert len(sizes) == 1, "every state must have the same tensor size"

    def test_state_json_round_trip(self, rig: Rig, sample_states: list[RigState]) -> None:
        for state in sample_states:
            data = json.loads(json.dumps(state.to_dict()))
            restored = rig.state_type.from_dict(data)
            assert restored.to_dict() == state.to_dict()
            assert torch.equal(restored.as_tensor(), state.as_tensor())

    # VJepaMapping

    def test_mapped_actions_shape(self, rig: Rig) -> None:
        mapped = rig.mapping.map_actions(rig.action_space.actions())
        assert mapped.shape == (len(rig.action_space), rig.mapping.action_dim)
        assert mapped.is_floating_point() and torch.isfinite(mapped).all()

    def test_mapping_is_batch_consistent(self, rig: Rig) -> None:
        actions = rig.action_space.actions()
        batched = rig.mapping.map_actions(actions)
        for action, row in zip(actions, batched):
            assert torch.allclose(rig.mapping.map_actions(action), row)

    def test_mapping_keeps_actions_distinct(self, rig: Rig) -> None:
        mapped = rig.mapping.map_actions(rig.action_space.actions())
        assert len(torch.unique(mapped, dim=0)) == len(mapped), "two actions map to the same input"

    def test_mapped_states_shape(self, rig: Rig, sample_states: list[RigState]) -> None:
        states = torch.stack([s.as_tensor() for s in sample_states])
        mapped = rig.mapping.map_states(states)
        assert mapped.shape == (len(sample_states), rig.mapping.state_dim)
        assert mapped.is_floating_point() and torch.isfinite(mapped).all()

    # CommandCodec

    def test_codec_action_round_trip(self, rig: Rig) -> None:
        for action in rig.action_space.actions():
            line = rig.codec.encode_action(action)
            assert isinstance(line, str) and line
            assert "\n" not in line and "\r" not in line
            decoded = rig.codec.decode_action(line)
            assert torch.allclose(decoded.to(action.dtype), action)
            assert rig.action_space.contains(decoded)

    def test_codec_state_round_trip(self, rig: Rig, sample_states: list[RigState]) -> None:
        for state in sample_states:
            line = rig.codec.encode_state(state)
            assert "\n" not in line and "\r" not in line
            decoded = rig.codec.decode_state(line)
            assert isinstance(decoded, rig.state_type)
            assert decoded.to_dict() == state.to_dict()

    def test_codec_rejects_garbage(self, rig: Rig) -> None:
        with pytest.raises(CodecError):
            rig.codec.decode_action("\x00not a command")
        with pytest.raises(CodecError):
            rig.codec.decode_state("\x00not a state")

    # Transport, end to end through the Rig

    def test_execute_every_action(self, open_rig: Rig) -> None:
        for action in open_rig.action_space.actions():
            assert isinstance(open_rig.execute(action), open_rig.state_type)

    def test_execute_rejects_foreign_actions(self, open_rig: Rig) -> None:
        with pytest.raises(ValueError):
            open_rig.execute(torch.full((open_rig.action_space.dim,), float("nan")))

    def test_close_closes_transport(self, rig: Rig) -> None:
        with rig:
            assert rig.transport.is_open
        assert not rig.transport.is_open

    # Camera

    def test_camera_capture(self, rig: Rig) -> None:
        if rig.camera is None:
            pytest.skip("rig has no camera")
        image = rig.camera.capture()
        height, width = rig.camera.resolution
        assert image.dtype == torch.uint8
        assert image.shape == (3, height, width)
