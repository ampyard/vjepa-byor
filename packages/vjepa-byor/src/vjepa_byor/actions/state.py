"""Rig state schema: a base class for states that are a few named numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Self

import torch


@dataclass(frozen=True)
class VectorState:
    """Base for rig states made of numeric fields; implements :class:`~vjepa_byor.interfaces.RigState`.

    Subclass it as a frozen dataclass; the fields, in declaration order, become
    the state tensor::

        @dataclass(frozen=True)
        class XYState(VectorState):
            x_mm: float
            y_mm: float
    """

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([float(getattr(self, name)) for name in self.field_names()])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        missing = [name for name in cls.field_names() if name not in data]
        if missing:
            raise ValueError(f"{cls.__name__} is missing {', '.join(missing)}")
        return cls(**{name: data[name] for name in cls.field_names()})

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        """Inverse of :meth:`as_tensor`; values come back as Python floats."""
        names = cls.field_names()
        if tensor.shape != (len(names),):
            raise ValueError(f"{cls.__name__} needs a tensor of shape ({len(names)},), got {tuple(tensor.shape)}")
        return cls(**dict(zip(names, tensor.tolist())))
