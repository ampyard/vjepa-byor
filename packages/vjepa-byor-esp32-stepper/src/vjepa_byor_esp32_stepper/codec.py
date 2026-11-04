"""The wire format spoken by the ESP32 firmware.

Plain text, one message per line, simple enough to parse on the microcontroller
without a JSON library. All distances are millimetres; the firmware converts
them to motor steps.

====================  =========  ===============================================
Message               Direction  Meaning
====================  =========  ===============================================
``MOVE <dx> <dy>``    to rig     relative move; whole millimetres, signed
``POS <x> <y>``       from rig   move finished; position, up to 3 decimals
``ERR <message>``     from rig   move refused or failed; position is unchanged
====================  =========  ===============================================

Example exchange: ``MOVE 5 -5`` -> ``POS 105 70``; ``MOVE 20 0`` at the edge ->
``ERR out of range``.
"""

from __future__ import annotations

import re

import torch

from vjepa_byor import CodecError, RigError
from vjepa_byor_esp32_stepper.actions import XYState

_INT = r"-?\d+"
_DECIMAL = r"-?\d+(?:\.\d{1,3})?"
_MOVE = re.compile(rf"MOVE ({_INT}) ({_INT})")
_POS = re.compile(rf"POS ({_DECIMAL}) ({_DECIMAL})")
_ERR = re.compile(r"ERR (.+)")


def format_mm(value: float) -> str:
    """Millimetres with at most 3 decimals and no trailing zeros: ``105``, ``12.5``, ``-0.125``."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


class Esp32Codec:
    """Implements :class:`~vjepa_byor.interfaces.CommandCodec` for the firmware's text protocol."""

    def encode_action(self, action: torch.Tensor) -> str:
        if action.shape != (2,):
            raise ValueError(f"expected an [dx, dy] action, got shape {tuple(action.shape)}")
        dx, dy = action.tolist()
        if not (float(dx).is_integer() and float(dy).is_integer()):
            raise ValueError(f"moves must be whole millimetres, got {[dx, dy]}")
        return f"MOVE {int(dx)} {int(dy)}"

    def decode_action(self, line: str) -> torch.Tensor:
        if not (m := _MOVE.fullmatch(line)):
            raise CodecError(f"bad command: {line!r}")
        return torch.tensor([float(m[1]), float(m[2])])

    def encode_state(self, state: XYState) -> str:
        return f"POS {format_mm(state.x_mm)} {format_mm(state.y_mm)}"

    def decode_state(self, line: str) -> XYState:
        if m := _ERR.fullmatch(line):
            raise RigError(m[1])
        if not (m := _POS.fullmatch(line)):
            raise CodecError(f"bad reply: {line!r}")
        return XYState(float(m[1]), float(m[2]))
