"""A top-down camera over the simulated stage, rendered with plain tensor ops.

The scene is a work mat with a 10 mm grid on a table, two fixed landmark blocks
(red top left, blue bottom right) and the carriage: an orange puck with a dark
centre dot and a soft shadow. Edges are anti-aliased, so the carriage's
sub-pixel position shows in the image and even a 1 mm move changes the photo.
Rendering is deterministic: the same position always gives the same photo.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vjepa_byor.sim.stage import SimStage

# V-JEPA 2-AC's input size; the camera service (backlog module 8) must match it
DEFAULT_RESOLUTION = (256, 256)

CARRIAGE_RADIUS_MM = 8.0
DOT_RADIUS_MM = 2.0
MIN_DOT_RADIUS_PX = 1.5
GRID_MM = 10.0
BLOCK_SIZE_MM = 14.0
PAD_PX = 4

TABLE = (0.55, 0.45, 0.36)
MAT = (0.84, 0.86, 0.82)
GRID = (0.62, 0.70, 0.72)
RED_BLOCK = (0.78, 0.16, 0.14)
BLUE_BLOCK = (0.15, 0.30, 0.72)
CARRIAGE = (0.95, 0.52, 0.10)
CARRIAGE_DOT = (0.20, 0.12, 0.08)

# Landmark centres as fractions of the travel area
RED_BLOCK_AT = (0.2, 0.75)
BLUE_BLOCK_AT = (0.8, 0.25)


def _blend(image: torch.Tensor, coverage: torch.Tensor, color: Sequence[float], alpha: float = 1.0) -> torch.Tensor:
    a = coverage * alpha
    return image * (1 - a) + torch.tensor(color).view(3, 1, 1) * a


class SimCamera:
    """Implements :class:`~vjepa_byor.interfaces.Camera` by drawing where ``stage``'s carriage is."""

    def __init__(self, stage: SimStage, resolution: Sequence[int] = DEFAULT_RESOLUTION) -> None:
        height, width = (int(v) for v in resolution)
        if min(height, width) < 32:
            raise ValueError(f"resolution must be at least 32 x 32, got {[height, width]}")
        self.stage = stage
        self._resolution = (height, width)

        # Fit the travel area, plus the carriage's radius all round, inside the frame
        r = CARRIAGE_RADIUS_MM
        span_w, span_h = stage.width_mm + 2 * r, stage.height_mm + 2 * r
        self.px_per_mm = min((width - 2 * PAD_PX) / span_w, (height - 2 * PAD_PX) / span_h)
        self._x0 = (width - span_w * self.px_per_mm) / 2 + r * self.px_per_mm
        self._y0 = height - ((height - span_h * self.px_per_mm) / 2 + r * self.px_per_mm)

        rows, cols = torch.meshgrid(torch.arange(height) + 0.5, torch.arange(width) + 0.5, indexing="ij")
        self._cols, self._rows = cols, rows
        self._background = self._render_background()

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """``(column, row)`` in pixels of a stage position; y points up, rows go down."""
        return self._x0 + x_mm * self.px_per_mm, self._y0 - y_mm * self.px_per_mm

    def _disc(self, x_mm: float, y_mm: float, radius_mm: float, min_radius_px: float = 0.0) -> torch.Tensor:
        cx, cy = self.to_pixel(x_mm, y_mm)
        distance = torch.hypot(self._cols - cx, self._rows - cy)
        return (max(radius_mm * self.px_per_mm, min_radius_px) - distance + 0.5).clamp(0, 1)

    def _rect(self, x0_mm: float, y0_mm: float, x1_mm: float, y1_mm: float) -> torch.Tensor:
        left, bottom = self.to_pixel(x0_mm, y0_mm)
        right, top = self.to_pixel(x1_mm, y1_mm)
        inside = torch.minimum(
            torch.minimum(self._cols - left, right - self._cols), torch.minimum(self._rows - top, bottom - self._rows)
        )
        return (inside + 0.5).clamp(0, 1)

    def _grid_lines(self) -> torch.Tensor:
        spacing = GRID_MM * self.px_per_mm
        def to_line(offset_px: torch.Tensor) -> torch.Tensor:
            phase = torch.remainder(offset_px / spacing, 1.0)
            return (0.5 - (phase - 0.5).abs()) * spacing

        nearest = torch.minimum(to_line(self._cols - self._x0), to_line(self._y0 - self._rows))  # px to a grid line
        return (1.0 - nearest).clamp(0, 1)

    def _render_background(self) -> torch.Tensor:
        height, width = self._resolution
        w, h, r = self.stage.width_mm, self.stage.height_mm, CARRIAGE_RADIUS_MM
        image = torch.tensor(TABLE).view(3, 1, 1).expand(3, height, width).clone()
        image = _blend(image, self._rect(-r, -r, w + r, h + r), MAT)
        image = _blend(image, self._grid_lines() * self._rect(0, 0, w, h), GRID)
        half = BLOCK_SIZE_MM / 2
        for (fx, fy), color in ((RED_BLOCK_AT, RED_BLOCK), (BLUE_BLOCK_AT, BLUE_BLOCK)):
            x, y = fx * w, fy * h
            image = _blend(image, self._rect(x - half + 1.5, y - half - 1.5, x + half + 1.5, y + half - 1.5), (0, 0, 0), 0.25)
            image = _blend(image, self._rect(x - half, y - half, x + half, y + half), color)
        return image

    def capture(self) -> torch.Tensor:
        x, y, r = self.stage.x_mm, self.stage.y_mm, CARRIAGE_RADIUS_MM
        image = _blend(self._background, self._disc(x + 1.5, y - 1.5, r), (0, 0, 0), 0.3)
        image = _blend(image, self._disc(x, y, r), CARRIAGE)
        image = _blend(image, self._disc(x, y, DOT_RADIUS_MM, MIN_DOT_RADIUS_PX), CARRIAGE_DOT)
        return (image * 255).round().clamp(0, 255).to(torch.uint8)
