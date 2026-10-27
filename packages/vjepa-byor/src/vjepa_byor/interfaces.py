"""Protocols every rig implements.

Conventions shared by all of them:

- An action is a 1-D float tensor of length ``ActionSpace.dim``. A batch of
  actions is a 2-D tensor of shape ``(N, dim)``.
- A rig state is the rig's position, reported by the hardware after every move.
- Wire messages are single lines of text without a trailing newline. Framing is
  the transport's job.

The core library contains no hardware code: rig plugins provide these pieces
and bundle them in a :class:`vjepa_byor.rig.Rig`.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

import torch


class CodecError(ValueError):
    """A wire message could not be decoded."""


class RigError(RuntimeError):
    """The rig reported a failure, such as an out-of-range move."""


@runtime_checkable
class RigState(Protocol):
    """The rig's position at one moment."""

    def as_tensor(self) -> torch.Tensor:
        """1-D float tensor; this is what ``VJepaMapping.map_states`` consumes."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, used in dataset manifests."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Inverse of :meth:`to_dict`."""
        ...


@runtime_checkable
class ActionSpace(Protocol):
    """A small, fixed set of actions: the basket the planner scores."""

    @property
    def dim(self) -> int:
        """Length of one action tensor."""
        ...

    def actions(self) -> torch.Tensor:
        """Every action as a float tensor of shape ``(len(self), dim)``, in a stable order."""
        ...

    def labels(self) -> list[str]:
        """A unique human-readable label for each row of :meth:`actions`."""
        ...

    def __len__(self) -> int: ...

    def contains(self, action: torch.Tensor) -> bool:
        """Whether ``action`` is one of :meth:`actions`."""
        ...


@runtime_checkable
class VJepaMapping(Protocol):
    """Maps rig actions and states to the predictor's input format.

    For V-JEPA 2-AC both are 7-D (e.g. an XY move becomes ``[dx, dy, 0, 0, 0, 0, 0]``).
    A mapping with other sizes pairs with custom input layers.
    """

    @property
    def action_dim(self) -> int: ...

    @property
    def state_dim(self) -> int: ...

    def map_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """``(..., ActionSpace.dim)`` -> ``(..., action_dim)``."""
        ...

    def map_states(self, states: torch.Tensor) -> torch.Tensor:
        """``(..., len(RigState.as_tensor()))`` -> ``(..., state_dim)``."""
        ...


@runtime_checkable
class CommandCodec(Protocol):
    """Converts actions to command lines for the hardware, and its replies back to states."""

    def encode_action(self, action: torch.Tensor) -> str: ...

    def decode_action(self, line: str) -> torch.Tensor:
        """Inverse of :meth:`encode_action`. Raises :class:`CodecError` if unparseable."""
        ...

    def encode_state(self, state: RigState) -> str: ...

    def decode_state(self, line: str) -> RigState:
        """Parse the rig's reply to a command.

        Raises :class:`RigError` if the reply reports a failure, and
        :class:`CodecError` if it cannot be parsed.
        """
        ...


@runtime_checkable
class Transport(Protocol):
    """Carries command lines to the hardware and replies back (WiFi, Bluetooth, serial, ...)."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    @property
    def is_open(self) -> bool: ...

    def send(self, line: str) -> None: ...

    def receive(self, timeout: float | None = None) -> str:
        """Next reply line. Raises :class:`TimeoutError` if none arrives in time."""
        ...


@runtime_checkable
class Camera(Protocol):
    """Takes the photos the predictor sees."""

    @property
    def resolution(self) -> tuple[int, int]:
        """``(height, width)`` of every captured image."""
        ...

    def capture(self) -> torch.Tensor:
        """An RGB image as a ``uint8`` tensor of shape ``(3, height, width)``."""
        ...
