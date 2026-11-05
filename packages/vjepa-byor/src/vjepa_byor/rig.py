"""The Rig bundle: groups one rig's pieces under a single registered name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch

from vjepa_byor.interfaces import (
    ActionSpace,
    ActionTransport,
    Camera,
    CommandCodec,
    RigState,
    Transport,
    VJepaMapping,
)


@dataclass(eq=False, kw_only=True)
class Rig:
    """Everything the core library needs to drive and model one rig.

    ``name`` is what users pass to ``--rig`` and must match the name the rig is
    registered under.

    The rig is driven in one of two ways:

    - with a ``codec`` and a line :class:`~vjepa_byor.interfaces.Transport`, for
      devices that speak text commands (serial, WiFi, Bluetooth);
    - with no codec and an :class:`~vjepa_byor.interfaces.ActionTransport`, which
      takes action tensors directly (ROS 2, vendor SDKs, ...).
    """

    name: str
    action_space: ActionSpace
    state_type: type[RigState]
    mapping: VJepaMapping
    transport: Transport | ActionTransport
    codec: CommandCodec | None = None
    camera: Camera | None = None
    reply_timeout: float = 10.0

    def __post_init__(self) -> None:
        if self.codec is None:
            if not isinstance(self.transport, ActionTransport):
                raise TypeError(
                    f"rig {self.name!r} has no codec, so its transport must be an ActionTransport "
                    "(with execute(action)); pass codec=... to use a line Transport"
                )
        elif not isinstance(self.transport, Transport):
            raise TypeError(
                f"rig {self.name!r} has a codec, so its transport must be a line Transport "
                "(with send(line) and receive()); drop the codec to use an ActionTransport"
            )

    @property
    def uses_lines(self) -> bool:
        """Whether the rig is driven with text command lines (a codec and a line transport)."""
        return self.codec is not None

    def open(self) -> Self:
        if not self.transport.is_open:
            self.transport.open()
        return self

    def close(self) -> None:
        if self.transport.is_open:
            self.transport.close()

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def command_for(self, action: torch.Tensor) -> str | None:
        """The command line ``action`` is sent as, or ``None`` for a rig without a codec."""
        return None if self.codec is None else self.codec.encode_action(action)

    def execute(self, action: torch.Tensor) -> RigState:
        """Send one action and wait for the rig to report its new state.

        Raises ``ValueError`` if the action is not in the action space, and
        :class:`~vjepa_byor.interfaces.RigError` if the rig rejects it.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"action {action.tolist()} is not in the {self.name!r} action space")
        if self.codec is None:
            return self.transport.execute(action, self.reply_timeout)  # type: ignore[union-attr]
        self.transport.send(self.codec.encode_action(action))  # type: ignore[union-attr]
        return self.codec.decode_state(self.transport.receive(self.reply_timeout))  # type: ignore[union-attr]
