import pytest

from vjepa_byor.testing import FakeTransport


def test_responder_replies_are_received_in_order():
    transport = FakeTransport(lambda line: f"ACK {line}")
    transport.open()
    transport.send("a")
    transport.send("b")
    assert transport.receive() == "ACK a"
    assert transport.receive() == "ACK b"
    assert transport.sent == ["a", "b"]


def test_queued_replies_without_responder():
    transport = FakeTransport()
    transport.open()
    transport.queue_reply("hello")
    transport.send("ping")
    assert transport.receive() == "hello"


def test_receive_times_out_when_nothing_queued():
    transport = FakeTransport()
    transport.open()
    with pytest.raises(TimeoutError):
        transport.receive(timeout=0.1)


def test_responder_returning_none_queues_nothing():
    transport = FakeTransport(lambda line: None)
    transport.open()
    transport.send("fire and forget")
    with pytest.raises(TimeoutError):
        transport.receive()


def test_closed_transport_refuses_io():
    transport = FakeTransport()
    with pytest.raises(ConnectionError):
        transport.send("x")
    with pytest.raises(ConnectionError):
        transport.receive()


def test_line_breaks_are_rejected():
    transport = FakeTransport()
    transport.open()
    with pytest.raises(ValueError):
        transport.send("two\nlines")
