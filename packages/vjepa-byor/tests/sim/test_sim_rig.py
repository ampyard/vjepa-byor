import pytest
import torch

from vjepa_byor import RigError, interfaces, load_rig
from vjepa_byor.testing.collector_conformance import CollectorConformanceTests
from vjepa_byor.testing.conformance import RigConformanceTests
from vjepa_byor.sim import SimCamera, SimRig, SimStage, XYState
from vjepa_byor.sim.camera import CARRIAGE_DOT


class TestSimRig(RigConformanceTests, CollectorConformanceTests):
    registered_name = "sim"

    @pytest.fixture
    def rig(self):
        return SimRig()

    @pytest.fixture
    def sample_states(self):
        return [XYState(0, 0), XYState(100, 75), XYState(12.5, 149.875)]


def test_satisfies_the_protocols():
    rig = SimRig()
    assert isinstance(rig.action_space, interfaces.ActionSpace)
    assert isinstance(rig.mapping, interfaces.VJepaMapping)
    assert isinstance(rig.codec, interfaces.CommandCodec)
    assert isinstance(rig.transport, interfaces.Transport)
    assert isinstance(rig.camera, interfaces.Camera)
    assert isinstance(XYState(1, 2), interfaces.RigState)


# Same mechanics as the real stage


def test_vocabulary_is_8_directions_by_3_step_sizes():
    space = SimRig().action_space
    assert len(space) == 24 and space.dim == 2
    assert space.labels()[:8] == ["N 1mm", "NE 1mm", "E 1mm", "SE 1mm", "S 1mm", "SW 1mm", "W 1mm", "NW 1mm"]
    assert space.by_label("SW 20mm").tolist() == [-20.0, -20.0]


def test_xy_move_maps_to_7d_in_metres():
    rig = SimRig()
    assert rig.mapping.map_actions(rig.action_space.by_label("SE 5mm")).tolist() == pytest.approx(
        [0.005, -0.005, 0, 0, 0, 0, 0]
    )
    assert rig.mapping.map_states(XYState(100, 75).as_tensor()).tolist() == pytest.approx([0.1, 0.075, 0, 0, 0, 0, 0])


def test_options_come_from_rig_config():
    rig = load_rig("sim", step_sizes_mm=[2, 10], travel_mm=[60, 40], start_mm=[0, 0], resolution=[96, 128])
    assert len(rig.action_space) == 16
    assert rig.camera.resolution == (96, 128)
    with rig:
        assert rig.execute(rig.action_space.by_label("NE 10mm")) == XYState(10, 10)


# Position reporting after each move


def test_reports_its_position_after_every_move():
    with SimRig() as rig:
        assert rig.execute(rig.action_space.by_label("NE 5mm")) == XYState(105, 80)
        assert rig.execute(rig.action_space.by_label("W 20mm")) == XYState(85, 80)
        assert rig.transport.sent == ['{"action": [5, 5]}', '{"action": [-20, 0]}']


def test_out_of_range_move_is_refused_and_position_kept():
    with SimRig(start_mm=(195, 75)) as rig:
        with pytest.raises(RigError, match="out of range"):
            rig.execute(rig.action_space.by_label("E 20mm"))
        assert rig.execute(rig.action_space.by_label("E 5mm")) == XYState(200, 75)


def test_stage_answers_garbage_with_an_error():
    assert SimStage()("MOVE 1 0") == '{"error": "bad command"}'


def test_start_must_be_inside_the_travel_area():
    with pytest.raises(ValueError, match="outside"):
        SimStage(start_mm=(201, 0))


# Camera


def test_photo_format_and_determinism():
    camera = SimRig().camera
    image = camera.capture()
    assert image.dtype == torch.uint8 and image.shape == (3, 256, 256)
    assert torch.equal(image, camera.capture())


@pytest.mark.parametrize("resolution", [(256, 256), (96, 128), (240, 160)])
@pytest.mark.parametrize("position", [(0, 0), (100, 75), (200, 150), (37, 121)])
def test_carriage_is_drawn_where_the_stage_is(resolution, position):
    stage = SimStage(start_mm=position)
    camera = SimCamera(stage, resolution)
    image = camera.capture()
    col, row = (int(v) for v in camera.to_pixel(*position))
    assert 0 <= row < resolution[0] and 0 <= col < resolution[1]
    expected = torch.tensor([round(c * 255) for c in CARRIAGE_DOT], dtype=torch.uint8)
    assert torch.equal(image[:, row, col], expected)


def test_carriage_never_leaves_the_frame():
    stage = SimStage()
    camera = SimCamera(stage, (64, 64))
    r = 8 * camera.px_per_mm
    for x, y in [(0, 0), (200, 0), (0, 150), (200, 150)]:
        col, row = camera.to_pixel(x, y)
        assert r <= col <= 64 - r and r <= row <= 64 - r


def test_every_1mm_move_changes_the_photo():
    stage = SimStage()
    camera = SimCamera(stage)
    for x, y in [(100, 75), (0, 0), (200, 150), (3, 148)]:
        stage.x_mm, stage.y_mm = x, y
        before = camera.capture()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if stage.in_range(x + dx, y + dy):
                stage.x_mm, stage.y_mm = x + dx, y + dy
                assert not torch.equal(camera.capture(), before), (x, y, dx, dy)
        stage.x_mm, stage.y_mm = x, y


def test_y_points_up_in_the_photo():
    camera = SimRig().camera
    col_a, row_a = camera.to_pixel(10, 10)
    col_b, row_b = camera.to_pixel(20, 20)
    assert col_b > col_a and row_b < row_a
