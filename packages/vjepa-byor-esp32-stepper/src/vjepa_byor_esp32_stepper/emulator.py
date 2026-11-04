"""A Python stand-in for the ESP32 firmware, for testing without hardware.

Pass it to :class:`vjepa_byor.testing.FakeTransport` as the responder::

    device = Esp32Emulator()
    rig = Esp32StepperRig(transport=FakeTransport(device))
"""

from __future__ import annotations

from vjepa_byor import CodecError
from vjepa_byor_esp32_stepper.codec import Esp32Codec, format_mm

DEFAULT_TRAVEL_MM = (200, 150)


class Esp32Emulator:
    """Answers ``MOVE`` commands like the firmware: refuses moves that would leave the travel area."""

    def __init__(
        self,
        travel_mm: tuple[float, float] = DEFAULT_TRAVEL_MM,
        x_mm: float | None = None,
        y_mm: float | None = None,
    ) -> None:
        self.width_mm, self.height_mm = travel_mm
        self.x_mm = self.width_mm / 2 if x_mm is None else x_mm
        self.y_mm = self.height_mm / 2 if y_mm is None else y_mm
        self._codec = Esp32Codec()

    def __call__(self, line: str) -> str:
        try:
            dx, dy = self._codec.decode_action(line).tolist()
        except CodecError:
            return "ERR bad command"
        x, y = self.x_mm + dx, self.y_mm + dy
        if not (0 <= x <= self.width_mm and 0 <= y <= self.height_mm):
            return "ERR out of range"
        self.x_mm, self.y_mm = x, y
        return f"POS {format_mm(self.x_mm)} {format_mm(self.y_mm)}"
