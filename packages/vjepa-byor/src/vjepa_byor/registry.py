"""Finds rig plugins by name.

Plugins register under the ``vjepa_byor.rigs`` entry-point group. The target is
either a :class:`Rig` or a callable (class or factory function) that returns
one; options passed to :func:`load_rig` go to that callable::

    [project.entry-points."vjepa_byor.rigs"]
    esp32-stepper = "vjepa_byor_esp32_stepper:Esp32StepperRig"

Rigs can also be registered in-process with :func:`register_rig`, which is
handy in notebooks and tests.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from vjepa_byor.rig import Rig

ENTRY_POINT_GROUP = "vjepa_byor.rigs"

RigFactory = Callable[..., Rig]

_registered: dict[str, Rig | RigFactory] = {}


class RigNotFoundError(LookupError):
    """No rig is registered under the requested name."""


def _entry_points() -> list[EntryPoint]:
    return list(entry_points(group=ENTRY_POINT_GROUP))


def _plugin_targets() -> dict[str, EntryPoint]:
    found: dict[str, EntryPoint] = {}
    for ep in _entry_points():
        if ep.name in found and found[ep.name].value != ep.value:
            raise RuntimeError(
                f"rig name {ep.name!r} is registered by more than one plugin: "
                f"{found[ep.name].value} and {ep.value}"
            )
        found[ep.name] = ep
    return found


def available_rigs() -> list[str]:
    """Names of every installed or in-process registered rig, sorted."""
    return sorted(set(_plugin_targets()) | set(_registered))


def register_rig(name: str, target: Rig | RigFactory, *, replace: bool = False) -> None:
    """Register a rig in-process, alongside the installed plugins."""
    if not replace and name in available_rigs():
        raise ValueError(f"a rig named {name!r} is already registered")
    _registered[name] = target


def unregister_rig(name: str) -> None:
    """Remove a rig registered with :func:`register_rig`."""
    _registered.pop(name, None)


def load_rig(name: str, **options: Any) -> Rig:
    """Build the rig registered as ``name``, passing ``options`` to its factory."""
    if name in _registered:
        target = _registered[name]
    else:
        plugins = _plugin_targets()
        if name not in plugins:
            raise RigNotFoundError(
                f"no rig named {name!r}; available: {', '.join(available_rigs()) or 'none'}"
            )
        target = plugins[name].load()

    if isinstance(target, Rig):
        if options:
            raise TypeError(f"rig {name!r} is a ready-made Rig and takes no options")
        rig = target
    elif callable(target):
        rig = target(**options)
    else:
        raise TypeError(f"rig {name!r} must be registered as a Rig or a callable returning one")

    if not isinstance(rig, Rig):
        raise TypeError(f"rig {name!r} factory returned {type(rig).__name__}, not Rig")
    if rig.name != name:
        raise ValueError(f"rig registered as {name!r} calls itself {rig.name!r}")
    return rig
