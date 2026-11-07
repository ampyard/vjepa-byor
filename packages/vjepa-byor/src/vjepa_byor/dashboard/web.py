"""The dashboard's web server: the page, a server-sent event stream, and a few commands.

    GET  /            the page (static/index.html)
    GET  /config      what the page needs to draw: state names, camera size, mat geometry
    GET  /events      server-sent events from the :class:`~vjepa_byor.dashboard.hub.Hub`
    GET  /runs        saved runs, newest first
    POST /goal        {"state": [x, y], "step_by_step": false}: send a goal to /reach_goal
    POST /next        step-by-step mode: make the move the planner is waiting on
    POST /cancel      cancel the running goal (one the dashboard sent)
    POST /reset       put the robot back at its start (if it has a reset service)
    POST /replay      {"run": name, "speed": 1.0}: play a saved run
    POST /live        stop a replay

Standard library only; serves on a background thread.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any

from vjepa_byor.dashboard.hub import Hub

KEEPALIVE_S = 10.0


class Commands:
    """What the page can ask for; the ROS side fills these in. Each returns a short message or raises."""

    def __init__(
        self,
        send_goal: Callable[[list[float], bool], str] | None = None,
        cancel: Callable[[], str] | None = None,
        reset: Callable[[], str] | None = None,
        next_step: Callable[[], str] | None = None,
    ) -> None:
        self.send_goal, self.cancel, self.reset, self.next_step = send_goal, cancel, reset, next_step


def resolve_host(host: str) -> str:
    """The one IPv4 address ``host`` (a name or an address) stands for; raises ``ValueError`` otherwise.

    The server listens on that address only, i.e. on the interface that has it.
    """
    try:
        addresses = sorted({a[4][0] for a in socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)})
    except socket.gaierror as e:
        raise ValueError(f"can't resolve {host!r}: {e.strerror}") from None
    if len(addresses) != 1:
        raise ValueError(f"{host!r} resolves to {', '.join(addresses)}; give one of these addresses instead")
    return addresses[0]


class DashboardServer:
    """Serves the page on ``host`` (a name or address resolving to one IPv4 address) only."""

    def __init__(self, hub: Hub, commands: Commands, config: dict[str, Any], host: str, port: int = 8080) -> None:
        self.hub, self.commands, self.config = hub, commands, config
        self._page = files("vjepa_byor.dashboard").joinpath("static/index.html")
        self.host, self.address = host, resolve_host(host)
        self.httpd = ThreadingHTTPServer((self.address, port), self._handler())
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="dashboard-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.hub.close()
        self.httpd.shutdown()
        self.httpd.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # quiet
                pass

            def _reply(self, status: int, body: bytes, content_type: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, data: Any, status: int = HTTPStatus.OK) -> None:
                self._reply(status, json.dumps(data).encode())

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._reply(HTTPStatus.OK, server._page.read_bytes(), "text/html; charset=utf-8")  # edits show on reload
                elif path == "/config":
                    self._json(server.config)
                elif path == "/runs":
                    self._json(server.hub.runs())
                elif path == "/events":
                    self._stream()
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            def _stream(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = server.hub.subscribe()
                try:
                    while True:
                        try:
                            event = q.get(timeout=KEEPALIVE_S)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        if event is None:
                            return
                        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    server.hub.unsubscribe(q)
                    self.close_connection = True

            def do_POST(self) -> None:
                path = self.path.split("?")[0]
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("expected a JSON object")
                    message = self._command(path, body)
                except KeyError as e:
                    self._json({"error": str(e).strip("'\"")}, HTTPStatus.NOT_FOUND)
                except (ValueError, TypeError, RuntimeError) as e:
                    self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                else:
                    self._json({"ok": True, "message": message})

            def _command(self, path: str, body: dict[str, Any]) -> str:
                c, hub = server.commands, server.hub
                if path == "/goal":
                    state = body.get("state")
                    if not isinstance(state, list) or not all(isinstance(v, (int, float)) for v in state):
                        raise ValueError("expected {\"state\": [numbers]}")
                    if c.send_goal is None:
                        raise RuntimeError("this dashboard can't send goals")
                    step_by_step = body.get("step_by_step", False)
                    if not isinstance(step_by_step, bool):
                        raise ValueError("step_by_step must be true or false")
                    hub.stop_replay()
                    return c.send_goal([float(v) for v in state], step_by_step)
                if path == "/next":
                    if c.next_step is None:
                        raise RuntimeError("this dashboard can't step runs")
                    return c.next_step()
                if path == "/cancel":
                    if c.cancel is None:
                        raise RuntimeError("this dashboard can't cancel goals")
                    return c.cancel()
                if path == "/reset":
                    if c.reset is None:
                        raise RuntimeError("this robot has no reset")
                    if hub.running:
                        raise RuntimeError("a run is going on")
                    return c.reset()
                if path == "/replay":
                    run = body.get("run")
                    if not isinstance(run, str):
                        raise ValueError("expected {\"run\": name}")
                    hub.replay(run, float(body.get("speed", 1.0)))
                    return f"replaying {run}"
                if path == "/live":
                    hub.stop_replay()
                    return "live"
                raise KeyError(f"no command {path}")

        return Handler
