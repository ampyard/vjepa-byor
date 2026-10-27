import pytest
import torch

from vjepa_byor import RigError


def test_execute_sends_command_and_returns_reported_state(line_rig):
    with line_rig:
        state = line_rig.execute(torch.tensor([5.0]))
    assert state.x == 15
    assert line_rig.transport.sent == ["MOVE 5"]


def test_execute_rejects_unknown_action_without_sending(line_rig):
    with line_rig, pytest.raises(ValueError):
        line_rig.execute(torch.tensor([3.0]))
    assert line_rig.transport.sent == []


def test_out_of_range_move_raises_rig_error(line_rig_factory):
    rig = line_rig_factory(start=18)
    with rig, pytest.raises(RigError, match="out of range"):
        rig.execute(torch.tensor([5.0]))


def test_context_manager_opens_and_closes_transport(line_rig):
    assert not line_rig.transport.is_open
    with line_rig:
        assert line_rig.transport.is_open
    assert not line_rig.transport.is_open
