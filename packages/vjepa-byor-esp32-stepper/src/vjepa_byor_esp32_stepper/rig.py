"""Builds the ESP32 stepper stage as a :class:`vjepa_byor.Rig`."""

from __future__ import annotations

from collections.abc import Sequence

from vjepa_byor import Camera, Rig, Transport
from vjepa_byor_esp32_stepper.actions import DEFAULT_STEP_SIZES_MM, XYState, xy_action_space
from vjepa_byor_esp32_stepper.codec import Esp32Codec
from vjepa_byor_esp32_stepper.mapping import xy_mapping

RIG_NAME = "esp32-stepper"


class NotConnectedTransport:
    """Placeholder until the WiFi / Bluetooth transport exists (backlog module 7); refuses to open."""

    is_open = False

    def open(self) -> None:
        raise ConnectionError(
            "the esp32-stepper rig has no transport yet (backlog module 7); "
            "pass transport=... to Esp32StepperRig, e.g. a FakeTransport(Esp32Emulator())"
        )

    def close(self) -> None:
        pass

    def send(self, line: str) -> None:
        raise ConnectionError("transport is not open")

    def receive(self, timeout: float | None = None) -> str:
        raise ConnectionError("transport is not open")


def Esp32StepperRig(
    step_sizes_mm: Sequence[int] = DEFAULT_STEP_SIZES_MM,
    transport: Transport | None = None,
    camera: Camera | None = None,
    reply_timeout: float = 10.0,
) -> Rig:
    """The XY stage: 8 directions x ``step_sizes_mm``, positions in millimetres.

    Registered as ``esp32-stepper``; ``--rig-config`` can set ``step_sizes_mm``
    and ``reply_timeout``. The camera comes with backlog module 8.
    """
    return Rig(
        name=RIG_NAME,
        action_space=xy_action_space(step_sizes_mm),
        state_type=XYState,
        mapping=xy_mapping(),
        codec=Esp32Codec(),
        transport=transport if transport is not None else NotConnectedTransport(),
        camera=camera,
        reply_timeout=reply_timeout,
    )
