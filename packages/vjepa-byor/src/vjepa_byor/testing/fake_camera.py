"""A Camera that draws where the rig is, so before and after photos differ when it moves."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


class MarkerCamera:
    """Renders a white square on black at the position ``position()`` returns.

    ``position`` returns one or two coordinates (x, or x and y), each mapped
    linearly from its ``(low, high)`` in ``bounds`` onto the image. Hook it to
    your device emulator, e.g. ``MarkerCamera(lambda: [device.x], [(0, 20)])``.
    Distinct positions give distinct images as long as the image is wider (and
    taller) than the number of positions.
    """

    def __init__(
        self,
        position: Callable[[], Sequence[float]],
        bounds: Sequence[tuple[float, float]],
        resolution: tuple[int, int] = (32, 48),
        marker: int = 3,
    ) -> None:
        if not 1 <= len(bounds) <= 2:
            raise ValueError("MarkerCamera draws one or two coordinates")
        self.position = position
        self.bounds = list(bounds)
        self._resolution = resolution
        self.marker = marker

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def _pixel(self, value: float, low: float, high: float, size: int) -> int:
        span = (high - low) or 1.0
        fraction = min(max((value - low) / span, 0.0), 1.0)
        return round(fraction * (size - self.marker))

    def capture(self) -> torch.Tensor:
        height, width = self._resolution
        coords = list(self.position())
        x = self._pixel(coords[0], *self.bounds[0], width)
        y = self._pixel(coords[1], *self.bounds[1], height) if len(self.bounds) == 2 else (height - self.marker) // 2
        image = torch.zeros((3, height, width), dtype=torch.uint8)
        image[:, y : y + self.marker, x : x + self.marker] = 255
        return image
