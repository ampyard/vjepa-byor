"""vjepa-byor — Bring Your Own Robot: fine-tune V-JEPA 2-AC to predict what your robot's actions will do."""

from vjepa_byor.interfaces import (
    ActionSpace,
    ActionTransport,
    Camera,
    CodecError,
    CommandCodec,
    RigError,
    RigState,
    Transport,
    VJepaMapping,
)
from vjepa_byor.registry import (
    RigNotFoundError,
    available_rigs,
    load_rig,
    register_rig,
    unregister_rig,
)
from vjepa_byor.rig import Rig

__all__ = [
    "ActionSpace",
    "ActionTransport",
    "Camera",
    "CodecError",
    "CommandCodec",
    "Rig",
    "RigError",
    "RigNotFoundError",
    "RigState",
    "Transport",
    "VJepaMapping",
    "available_rigs",
    "load_rig",
    "register_rig",
    "unregister_rig",
]
