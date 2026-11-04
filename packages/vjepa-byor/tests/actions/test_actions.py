from dataclasses import dataclass

import pytest
import torch

from vjepa_byor import CodecError, Rig, RigError, interfaces
from vjepa_byor.actions import COMPASS, DiscreteActionSpace, JsonLineCodec, SlotMapping, VectorState
from vjepa_byor.actions.vjepa_mapping import X, Y, Z
from vjepa_byor.testing import FakeTransport
from vjepa_byor.testing.conformance import RigConformanceTests


@dataclass(frozen=True)
class PanTilt(VectorState):
    pan: float
    tilt: float


# DiscreteActionSpace


def test_compass_is_8_directions_by_step_size():
    space = DiscreteActionSpace.compass([1, 5, 20], unit="mm")
    assert len(space) == 24 and space.dim == 2
    assert space.labels()[:8] == [f"{d} 1mm" for d in COMPASS]
    assert space.by_label("NE 5mm").tolist() == [5.0, 5.0]
    assert space.by_label("W 20mm").tolist() == [-20.0, 0.0]


def test_index_of_and_contains():
    space = DiscreteActionSpace.compass([2])
    assert space.index_of(torch.tensor([0.0, 2.0])) == 0
    assert space.index_of(torch.tensor([0, -2])) == 4  # integer tensors match too
    assert space.index_of(torch.tensor([1.0, 1.0])) is None
    assert not space.contains(torch.tensor([0.0, 2.0, 0.0]))
    assert not space.contains(torch.tensor([[0.0, 2.0]]))


def test_actions_are_a_copy():
    space = DiscreteActionSpace.compass([1])
    space.actions()[0] = 99
    assert space.actions()[0].tolist() == [0.0, 1.0]


def test_by_label_unknown():
    with pytest.raises(KeyError):
        DiscreteActionSpace.compass([1]).by_label("up")


@pytest.mark.parametrize(
    ("rows", "labels"),
    [
        ([], []),
        ([[1.0], [2.0]], ["a"]),
        ([[1.0], [2.0]], ["a", "a"]),
        ([[1.0], [1.0]], ["a", "b"]),
        ([[float("inf")]], ["a"]),
    ],
)
def test_rejects_bad_vocabularies(rows, labels):
    with pytest.raises(ValueError):
        DiscreteActionSpace(rows, labels)


def test_compass_rejects_non_positive_steps():
    with pytest.raises(ValueError):
        DiscreteActionSpace.compass([0, 1])


# VectorState


def test_vector_state_round_trips():
    state = PanTilt(10.5, -3.0)
    assert state.as_tensor().tolist() == [10.5, -3.0]
    assert state.to_dict() == {"pan": 10.5, "tilt": -3.0}
    assert PanTilt.from_dict(state.to_dict()) == state
    assert PanTilt.from_tensor(state.as_tensor()) == state
    assert isinstance(state, interfaces.RigState)


def test_vector_state_errors():
    with pytest.raises(ValueError, match="tilt"):
        PanTilt.from_dict({"pan": 1})
    with pytest.raises(ValueError):
        PanTilt.from_tensor(torch.zeros(3))


# SlotMapping


def test_slot_mapping_places_and_scales():
    mapping = SlotMapping([X, Y], [X, Y], action_scale=1e-3, state_scale=1e-3, state_offset=[100, 50])
    assert mapping.map_actions(torch.tensor([5.0, -5.0])).tolist() == pytest.approx([0.005, -0.005, 0, 0, 0, 0, 0])
    assert mapping.map_states(torch.tensor([[100.0, 60.0]])).tolist() == [pytest.approx([0, 0.01, 0, 0, 0, 0, 0])]


def test_slot_mapping_other_slots_and_dtypes():
    mapping = SlotMapping([Z], [Y, Z], dim=4)
    assert mapping.action_dim == mapping.state_dim == 4
    assert mapping.map_actions(torch.tensor([3])).tolist() == [0.0, 0.0, 3.0, 0.0]
    assert mapping.map_actions(torch.tensor([3.0], dtype=torch.bfloat16)).dtype == torch.bfloat16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_slots": [], "state_slots": [0]},
        {"action_slots": [0, 0], "state_slots": [0]},
        {"action_slots": [7], "state_slots": [0]},
        {"action_slots": [0], "state_slots": [0], "action_scale": 0},
        {"action_slots": [0], "state_slots": [0, 1], "state_scale": [1, 2, 3]},
    ],
)
def test_slot_mapping_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        SlotMapping(**kwargs)


def test_slot_mapping_rejects_wrong_width():
    with pytest.raises(ValueError):
        SlotMapping([X, Y], [X, Y]).map_actions(torch.zeros(3))


# JsonLineCodec


def test_json_codec_wire_format():
    codec = JsonLineCodec(PanTilt, action_dim=2)
    assert codec.encode_action(torch.tensor([5.0, -0.5])) == '{"action": [5, -0.5]}'
    assert codec.decode_action('{"action": [5, -0.5]}').tolist() == [5.0, -0.5]
    assert codec.decode_state('{"state": {"pan": 1, "tilt": 2}}') == PanTilt(1, 2)


@pytest.mark.parametrize(
    "line",
    ['{"action": [1]}', '{"action": [1, "2"]}', '{"action": [true, 1]}', '{"action": [NaN, 1]}', "[1, 2]", ""],
)
def test_json_codec_rejects_bad_commands(line):
    with pytest.raises(CodecError):
        JsonLineCodec(PanTilt, 2).decode_action(line)


@pytest.mark.parametrize("line", ['{"state": {"pan": 1}}', '{"state": [1, 2]}', '{"pos": 1}'])
def test_json_codec_rejects_bad_replies(line):
    with pytest.raises(CodecError):
        JsonLineCodec(PanTilt, 2).decode_state(line)


def test_json_codec_error_reply():
    with pytest.raises(RigError, match="limit"):
        JsonLineCodec(PanTilt, 2).decode_state('{"error": "limit"}')


# A rig assembled only from the generic pieces passes the conformance suite


class PanTiltDevice:
    def __init__(self):
        self.state = PanTilt(0.0, 0.0)
        self.codec = JsonLineCodec(PanTilt, 2)

    def __call__(self, line):
        dpan, dtilt = self.codec.decode_action(line).tolist()
        if abs(self.state.tilt + dtilt) > 45:
            return '{"error": "tilt limit"}'
        self.state = PanTilt(self.state.pan + dpan, self.state.tilt + dtilt)
        return self.codec.encode_state(self.state)


class TestGenericPartsRig(RigConformanceTests):
    @pytest.fixture
    def rig(self):
        return Rig(
            name="pan-tilt",
            action_space=DiscreteActionSpace.compass([1, 5], unit="deg"),
            state_type=PanTilt,
            mapping=SlotMapping([X, Y], [X, Y], action_scale=0.01, state_scale=0.01),
            codec=JsonLineCodec(PanTilt, 2),
            transport=FakeTransport(PanTiltDevice()),
        )

    @pytest.fixture
    def sample_states(self):
        return [PanTilt(0.0, 0.0), PanTilt(-12.5, 45.0)]
