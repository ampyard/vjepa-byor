import pytest
import torch

from vjepa_byor import Rig, RigError
from vjepa_byor.testing import FakeActionTransport, FakeTransport


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


def test_line_rig_reports_its_command_lines(line_rig):
    assert line_rig.uses_lines
    assert line_rig.command_for(torch.tensor([5.0])) == "MOVE 5"


# Driven with action tensors through an ActionTransport


def test_direct_rig_passes_the_action_tensor_and_returns_the_state(direct_rig):
    with direct_rig:
        state = direct_rig.execute(torch.tensor([5.0]))
    assert state.x == 15
    assert [a.tolist() for a in direct_rig.transport.sent] == [[5.0]]


def test_direct_rig_has_no_command_lines(direct_rig):
    assert not direct_rig.uses_lines
    assert direct_rig.command_for(torch.tensor([5.0])) is None


def test_direct_rig_rejects_unknown_action_without_sending(direct_rig):
    with direct_rig, pytest.raises(ValueError):
        direct_rig.execute(torch.tensor([3.0]))
    assert direct_rig.transport.sent == []


def test_direct_rig_out_of_range_move_raises_rig_error(direct_rig_factory):
    rig = direct_rig_factory(start=18)
    with rig, pytest.raises(RigError, match="out of range"):
        rig.execute(torch.tensor([5.0]))
    assert rig.transport.responder.x == 18


def test_direct_rig_times_out_when_no_state_comes_back(direct_rig):
    direct_rig.transport.responder = lambda action: None
    with direct_rig, pytest.raises(TimeoutError):
        direct_rig.execute(torch.tensor([1.0]))


def test_direct_rig_passes_its_reply_timeout(direct_rig):
    seen = []
    transport = direct_rig.transport
    original = transport.execute
    transport.execute = lambda action, timeout=None: seen.append(timeout) or original(action, timeout)
    direct_rig.reply_timeout = 2.5
    with direct_rig:
        direct_rig.execute(torch.tensor([1.0]))
    assert seen == [2.5]


def test_direct_rig_opens_and_closes_its_transport(direct_rig):
    with direct_rig:
        assert direct_rig.transport.is_open
    assert not direct_rig.transport.is_open


def test_closed_action_transport_refuses_moves(direct_rig):
    with pytest.raises(ConnectionError):
        direct_rig.execute(torch.tensor([1.0]))


def _parts(rig):
    return {"name": rig.name, "action_space": rig.action_space, "state_type": rig.state_type, "mapping": rig.mapping}


def test_a_codec_needs_a_line_transport(line_rig, direct_rig):
    with pytest.raises(TypeError, match="line Transport"):
        Rig(**_parts(line_rig), codec=line_rig.codec, transport=FakeActionTransport(lambda action: None))


def test_no_codec_needs_an_action_transport(line_rig):
    with pytest.raises(TypeError, match="ActionTransport"):
        Rig(**_parts(line_rig), transport=FakeTransport())
