from importlib.metadata import EntryPoint

import pytest

from vjepa_byor import RigNotFoundError, available_rigs, load_rig, register_rig, registry, unregister_rig


class FakeEntryPoint:
    def __init__(self, name, target, value=None):
        self.name = name
        self.value = value or f"fake_module:{name}"
        self._target = target

    def load(self):
        return self._target


@pytest.fixture
def plugins(monkeypatch):
    """Replace installed entry points with an editable list."""
    installed: list[EntryPoint] = []
    monkeypatch.setattr(registry, "_entry_points", lambda: installed)
    monkeypatch.setattr(registry, "_registered", {})
    return installed


def test_discovers_plugins_from_entry_points(plugins, line_rig_factory):
    plugins.append(FakeEntryPoint("line", line_rig_factory))
    assert available_rigs() == ["line"]
    assert load_rig("line").name == "line"


def test_options_are_passed_to_the_factory(plugins, line_rig_factory):
    plugins.append(FakeEntryPoint("line", line_rig_factory))
    rig = load_rig("line", start=3)
    assert rig.transport.responder.x == 3


def test_ready_made_rig_is_returned_as_is(plugins, line_rig):
    plugins.append(FakeEntryPoint("line", line_rig))
    assert load_rig("line") is line_rig
    with pytest.raises(TypeError):
        load_rig("line", start=3)


def test_unknown_name_lists_available_rigs(plugins, line_rig_factory):
    plugins.append(FakeEntryPoint("line", line_rig_factory))
    with pytest.raises(RigNotFoundError, match="available: line"):
        load_rig("nope")


def test_conflicting_plugins_are_reported(plugins, line_rig_factory):
    plugins.append(FakeEntryPoint("line", line_rig_factory, "pkg_a:make"))
    plugins.append(FakeEntryPoint("line", line_rig_factory, "pkg_b:make"))
    with pytest.raises(RuntimeError, match="more than one plugin"):
        available_rigs()


def test_factory_must_return_a_rig(plugins):
    plugins.append(FakeEntryPoint("broken", lambda: object()))
    with pytest.raises(TypeError, match="not Rig"):
        load_rig("broken")


def test_rig_name_must_match_registered_name(plugins, line_rig_factory):
    plugins.append(FakeEntryPoint("other", line_rig_factory))
    with pytest.raises(ValueError, match="calls itself 'line'"):
        load_rig("other")


def test_in_process_registration(plugins, line_rig_factory):
    register_rig("line", line_rig_factory)
    assert available_rigs() == ["line"]
    assert load_rig("line").name == "line"
    with pytest.raises(ValueError, match="already registered"):
        register_rig("line", line_rig_factory)
    unregister_rig("line")
    assert available_rigs() == []


def test_in_process_registration_cannot_shadow_a_plugin_by_accident(plugins, line_rig_factory):
    plugins.append(FakeEntryPoint("line", line_rig_factory))
    with pytest.raises(ValueError):
        register_rig("line", line_rig_factory)
    register_rig("line", line_rig_factory, replace=True)
    assert load_rig("line").name == "line"
