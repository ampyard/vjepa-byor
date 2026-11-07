"""The dashboard without ROS: the event hub (snapshot, recording, replay) and the web server."""

import json
import queue
import threading
import time
import urllib.error
import urllib.request

import pytest

from vjepa_byor.dashboard import Commands, DashboardServer, Hub


def drain(q, timeout=0.0):
    events = []
    end = time.monotonic() + timeout
    while True:
        try:
            events.append(q.get(timeout=max(0.0, end - time.monotonic())))
        except queue.Empty:
            return events


def a_run(hub, reason="reached"):
    hub.publish({"type": "camera", "src": "cam-0"})
    hub.publish({"type": "goal_image", "src": "goal"})
    hub.publish({"type": "start", "goal_state": [120, 60], "threshold": 0.165})
    hub.publish({"type": "step", "step": 0, "chosen": "SE 20mm"})
    hub.publish({"type": "camera", "src": "cam-1"})
    hub.publish({"type": "state", "state": {"x_mm": 120, "y_mm": 55}})
    hub.publish({"type": "step", "step": 1, "chosen": None})
    hub.publish({"type": "end", "reason": reason, "reached": reason == "reached"})


def types(events):
    return [(e["type"], e.get("src") or e.get("step") or e.get("mode") or e.get("reason")) for e in events]


def test_live_events_reach_every_browser(tmp_path):
    hub = Hub(tmp_path)
    a, b = hub.subscribe(), hub.subscribe()
    drain(a), drain(b)
    hub.publish({"type": "camera", "src": "x"})
    assert drain(a) == drain(b) == [{"type": "camera", "src": "x"}]


def test_a_new_browser_gets_the_current_picture_and_the_last_run(tmp_path):
    hub = Hub(tmp_path)
    a_run(hub)
    snapshot = drain(hub.subscribe())
    assert types(snapshot) == [("mode", "live"), ("camera", "cam-1"), ("goal_image", "goal"), ("start", None),
                               ("step", None), ("step", 1), ("end", "reached"), ("state", None)]


def test_each_run_is_saved_with_its_photos_and_timing(tmp_path):
    t = iter(range(100))
    hub = Hub(tmp_path, clock=lambda: float(next(t)))
    assert not hub.running
    a_run(hub, reason="cycle")
    assert not hub.running
    [name] = hub.runs()
    assert name.endswith("_cycle")
    recording = hub.load(name)
    # the photo and goal photo from before the start, then everything but the state
    assert types(e for _, e in recording) == [("camera", "cam-0"), ("goal_image", "goal"), ("start", None),
                                             ("step", None), ("camera", "cam-1"), ("step", 1), ("end", "cycle")]
    times = [t for t, _ in recording]
    assert times == sorted(times) and times[0] == 0.0


def test_runs_are_listed_newest_first_and_names_never_clash(tmp_path):
    hub = Hub(tmp_path)
    a_run(hub)
    a_run(hub)
    names = hub.runs()
    assert len(names) == 2 and len(set(names)) == 2
    with pytest.raises(KeyError):
        hub.load("../etc/passwd")


def test_replay_plays_a_run_back_and_holds_live_events(tmp_path):
    hub = Hub(tmp_path)
    a_run(hub)
    [name] = hub.runs()
    q = hub.subscribe()
    drain(q)
    hub.replay(name, speed=100)
    assert hub.replaying
    hub.publish({"type": "camera", "src": "live-while-replaying"})
    played = drain(q, timeout=0.5)
    assert ("camera", "live-while-replaying") not in types(played)
    assert types(played)[0] == ("mode", "replay")
    assert types(played)[1:] == [("camera", "cam-0"), ("goal_image", "goal"), ("start", None), ("step", None),
                                 ("camera", "cam-1"), ("step", 1), ("end", "reached")]
    hub.stop_replay()
    back = drain(q)
    assert types(back)[:2] == [("mode", "live"), ("camera", "live-while-replaying")]


def test_a_live_run_ends_a_replay_and_no_replay_during_a_run(tmp_path):
    hub = Hub(tmp_path)
    a_run(hub)
    [name] = hub.runs()
    hub.replay(name, speed=0.01)
    hub.publish({"type": "start", "goal_state": [1, 2]})
    assert not hub.replaying and hub.running
    with pytest.raises(RuntimeError, match="live run"):
        hub.replay(name)


# The web server


@pytest.fixture
def web(tmp_path):
    hub = Hub(tmp_path)
    sent = []
    commands = Commands(send_goal=lambda s, step: sent.append((s, step)) or "goal sent", cancel=lambda: "cancelling",
                        next_step=lambda: "moving")
    server = DashboardServer(hub, commands, {"state_names": ["x_mm", "y_mm"]}, host="127.0.0.1", port=0)
    server.start()
    server.sent = sent
    yield server
    server.stop()


def get(server, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}", timeout=5) as r:
        return r.status, r.headers["Content-Type"], r.read()


def post(server, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serves_the_page_and_the_config(web):
    status, kind, body = get(web, "/")
    assert status == 200 and kind.startswith("text/html") and b"reach goal" in body
    assert json.loads(get(web, "/config")[2]) == {"state_names": ["x_mm", "y_mm"]}
    assert json.loads(get(web, "/runs")[2]) == []


def test_goal_command(web):
    assert post(web, "/goal", {"state": [120, 60]}) == (200, {"ok": True, "message": "goal sent"})
    assert post(web, "/goal", {"state": [90, 80], "step_by_step": True})[0] == 200
    assert web.sent == [([120.0, 60.0], False), ([90.0, 80.0], True)]
    assert post(web, "/goal", {"state": [90, 80], "step_by_step": "yes"})[0] == 400
    assert post(web, "/next", {}) == (200, {"ok": True, "message": "moving"})
    assert post(web, "/goal", {"state": "here"})[0] == 400
    assert post(web, "/reset", {}) == (400, {"error": "this robot has no reset"})
    assert post(web, "/nope", {})[0] == 404
    assert post(web, "/replay", {"run": "missing"})[0] == 404


def test_event_stream_starts_with_the_snapshot_then_live_events(web):
    web.hub.publish({"type": "camera", "src": "cam"})
    lines = []

    def read():
        with urllib.request.urlopen(f"http://127.0.0.1:{web.port}/events", timeout=5) as r:
            for raw in r:
                if raw.startswith(b"data: "):
                    lines.append(json.loads(raw[6:]))
                if len(lines) == 3:
                    return

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    end = time.monotonic() + 5
    while len(lines) < 2 and time.monotonic() < end:
        time.sleep(0.01)
    web.hub.publish({"type": "state", "state": {"x_mm": 1, "y_mm": 2}})
    reader.join(timeout=5)
    assert types(lines) == [("mode", "live"), ("camera", "cam"), ("state", None)]


def test_listens_only_on_the_address_the_host_name_resolves_to(tmp_path):
    from vjepa_byor.dashboard.web import resolve_host

    assert resolve_host("127.0.0.1") == "127.0.0.1"
    assert resolve_host("localhost") == "127.0.0.1"
    with pytest.raises(ValueError, match="can't resolve"):
        resolve_host("no-such-host.invalid")
    server = DashboardServer(Hub(tmp_path), Commands(), {}, host="localhost", port=0)
    try:
        assert server.httpd.server_address[0] == "127.0.0.1"
    finally:
        server.httpd.server_close()


def test_clear_forgets_the_last_run_but_never_a_running_one(tmp_path):
    hub = Hub(tmp_path)
    a_run(hub)
    q = hub.subscribe()
    drain(q)
    hub.clear()
    assert types(drain(q)) == [("clear", None)]
    snapshot = types(drain(hub.subscribe()))
    assert ("start", None) not in snapshot and ("goal_image", "goal") not in snapshot
    hub.publish({"type": "start"})
    hub.clear()
    assert ("start", None) in types(drain(hub.subscribe()))


def test_pending_decisions_are_part_of_the_run(tmp_path):
    hub = Hub(tmp_path)
    hub.publish({"type": "start"})
    hub.publish({"type": "pending", "step": 0, "planned": "SE 20mm"})
    assert ("pending", None) in types(drain(hub.subscribe()))
    hub.publish({"type": "step", "step": 0})
    hub.publish({"type": "end", "reason": "reached"})
    assert [e["type"] for _, e in hub.load(hub.runs()[0])] == ["start", "pending", "step", "end"]
