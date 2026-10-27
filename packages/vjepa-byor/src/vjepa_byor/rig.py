"""The Rig bundle: groups one rig's pieces under a single registered name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch

from vjepa_byor.interfaces import (
    ActionSpace,
    Camera,
    CommandCodec,
    RigState,
    Transport,
    VJepaMapping,
)


@dataclass(eq=False)
class Rig:
    """Everything the core library needs to drive and model one rig.

    ``name`` is what users pass to ``--rig`` and must match the name the rig is
    registered under.
    """

    name: str
    action_space: ActionSpace
    state_type: type[RigState]
    mapping: VJepaMapping
    codec: CommandCodec
    transport: Transport
    camera: Camera | None = None
    reply_timeout: float = 10.0

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

    def execute(self, action: torch.Tensor) -> RigState:
        """Send one action and wait for the rig to report its new state.

        Raises ``ValueError`` if the action is not in the action space, and
        :class:`~vjepa_byor.interfaces.RigError` if the rig rejects it.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"action {action.tolist()} is not in the {self.name!r} action space")
        self.transport.send(self.codec.encode_action(action))
        return self.codec.decode_state(self.transport.receive(self.reply_timeout))
