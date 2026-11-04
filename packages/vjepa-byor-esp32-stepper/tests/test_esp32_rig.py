import pytest
import torch

from vjepa_byor import CodecError, RigError, interfaces, load_rig
from vjepa_byor.testing import FakeTransport
from vjepa_byor.testing.collector_conformance import CollectorConformanceTests
from vjepa_byor.testing.conformance import RigConformanceTests
from vjepa_byor.testing.fake_camera import MarkerCamera
from vjepa_byor_esp32_stepper import Esp32Codec, Esp32Emulator, Esp32StepperRig, XYState, xy_action_space


def emulated_rig(**emulator_options):
    device = Esp32Emulator(**emulator_options)
    camera = MarkerCamera(
        lambda: [device.x_mm, device.y_mm], [(0, device.width_mm), (0, device.height_mm)], resolution=(160, 208)
    )
    return Esp32StepperRig(transport=FakeTransport(device), camera=camera)


class TestEsp32StepperRig(RigConformanceTests, CollectorConformanceTests):
    registered_name = "esp32-stepper"

    @pytest.fixture
    def rig(self):
        return emulated_rig()

    @pytest.fixture
    def sample_states(self):
        return [XYState(0, 0), XYState(100, 75), XYState(12.5, 149.875)]


# Action vocabulary


def test_vocabulary_is_8_directions_by_3_step_sizes():
    space = xy_action_space()
    assert len(space) == 24 and space.dim == 2
    assert space.labels()[:8] == ["N 1mm", "NE 1mm", "E 1mm", "SE 1mm", "S 1mm", "SW 1mm", "W 1mm", "NW 1mm"]
    assert space.by_label("SW 20mm").tolist() == [-20.0, -20.0]


def test_step_sizes_must_be_whole_millimetres():
    with pytest.raises(ValueError):
        xy_action_space([1, 2.5])


def test_step_sizes_come_from_rig_options():
    rig = load_rig("esp32-stepper", step_sizes_mm=[2, 10])
    assert len(rig.action_space) == 16
    assert rig.action_space.by_label("E 10mm").tolist() == [10.0, 0.0]


# State


def test_state_fields_are_floats():
    state = XYState.from_dict({"x_mm": 3, "y_mm": 4})
    assert state == XYState(3.0, 4.0)
    assert isinstance(state.x_mm, float)
    assert isinstance(state, interfaces.RigState)


# V-JEPA mapping


def test_xy_move_maps_to_7d_in_metres():
    rig = Esp32StepperRig()
    mapped = rig.mapping.map_actions(rig.action_space.by_label("SE 5mm"))
    assert mapped.tolist() == pytest.approx([0.005, -0.005, 0, 0, 0, 0, 0])
    assert rig.mapping.map_states(XYState(100, 75).as_tensor()).tolist() == pytest.approx([0.1, 0.075, 0, 0, 0, 0, 0])


# Wire format


@pytest.mark.parametrize(
    ("label", "line"), [("N 1mm", "MOVE 0 1"), ("SW 20mm", "MOVE -20 -20"), ("E 5mm", "MOVE 5 0")]
)
def test_command_lines(label, line):
    codec = Esp32Codec()
    action = xy_action_space().by_label(label)
    assert codec.encode_action(action) == line
    assert torch.equal(codec.decode_action(line), action)


@pytest.mark.parametrize(
    ("state", "line"),
    [(XYState(105, 70), "POS 105 70"), (XYState(12.5, -0.125), "POS 12.5 -0.125"), (XYState(-0.0, 0), "POS 0 0")],
)
def test_state_lines(state, line):
    codec = Esp32Codec()
    assert codec.encode_state(state) == line
    assert codec.decode_state(line) == state


def test_error_reply_raises_rig_error():
    with pytest.raises(RigError, match="out of range"):
        Esp32Codec().decode_state("ERR out of range")


@pytest.mark.parametrize("line", ["MOVE 1", "MOVE 1.5 0", "move 1 0", "MOVE 1 0 ", "MOVE  1 0", ""])
def test_rejects_bad_commands(line):
    with pytest.raises(CodecError):
        Esp32Codec().decode_action(line)


@pytest.mark.parametrize("line", ["POS 1", "POS 1.2345 0", "POS a b", "OK", "ERR", ""])
def test_rejects_bad_replies(line):
    with pytest.raises(CodecError):
        Esp32Codec().decode_state(line)


def test_refuses_to_encode_fractional_moves():
    with pytest.raises(ValueError):
        Esp32Codec().encode_action(torch.tensor([0.5, 0.0]))


# Through the Rig, against the firmware emulator


def test_execute_moves_the_carriage():
    with emulated_rig() as rig:
        state = rig.execute(rig.action_space.by_label("NE 5mm"))
        assert state == XYState(105, 80)
        assert rig.transport.sent == ["MOVE 5 5"]


def test_out_of_range_move_is_refused_and_position_kept():
    with emulated_rig(x_mm=195, y_mm=75) as rig:
        with pytest.raises(RigError, match="out of range"):
            rig.execute(rig.action_space.by_label("E 20mm"))
        assert rig.execute(rig.action_space.by_label("E 5mm")) == XYState(200, 75)


def test_emulator_rejects_garbage():
    assert Esp32Emulator()("JUMP") == "ERR bad command"


def test_default_transport_explains_it_is_not_there_yet():
    rig = load_rig("esp32-stepper")
    with pytest.raises(ConnectionError, match="module 7"):
        rig.open()
