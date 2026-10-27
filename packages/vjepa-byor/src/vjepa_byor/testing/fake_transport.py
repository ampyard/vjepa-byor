"""An in-memory Transport: no hardware, no sockets."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable


class FakeTransport:
    """Records every line sent and serves replies from a queue.

    ``responder`` stands in for the device: it is called with each sent line
    and its return value (unless ``None``) is queued as the reply. Replies can
    also be queued by hand with :meth:`queue_reply`.
    """

    def __init__(self, responder: Callable[[str], str | None] | None = None) -> None:
        self.responder = responder
        self.sent: list[str] = []
        self._replies: deque[str] = deque()
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def send(self, line: str) -> None:
        if not self._open:
            raise ConnectionError("transport is not open")
        if "\n" in line or "\r" in line:
            raise ValueError(f"command lines must not contain line breaks: {line!r}")
        self.sent.append(line)
        if self.responder is not None:
            reply = self.responder(line)
            if reply is not None:
                self._replies.append(reply)

    def receive(self, timeout: float | None = None) -> str:
        if not self._open:
            raise ConnectionError("transport is not open")
        if not self._replies:
            raise TimeoutError("no reply queued")
        return self._replies.popleft()

    def queue_reply(self, line: str) -> None:
        self._replies.append(line)
