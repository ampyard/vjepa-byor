"""Generic building blocks for a rig's actions, state, V-JEPA mapping and wire format.

Rig plugins combine these with their own settings; none of them is specific to
one piece of hardware.
"""

from vjepa_byor.actions.codecs import JsonLineCodec
from vjepa_byor.actions.state import VectorState
from vjepa_byor.actions.vjepa_mapping import VJEPA_AC_DIM, SlotMapping
from vjepa_byor.actions.vocabulary import COMPASS, DiscreteActionSpace

__all__ = [
    "COMPASS",
    "VJEPA_AC_DIM",
    "DiscreteActionSpace",
    "JsonLineCodec",
    "SlotMapping",
    "VectorState",
]
