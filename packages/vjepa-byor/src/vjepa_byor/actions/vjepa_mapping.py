"""Maps rig actions and states into V-JEPA 2-AC's 7-D action/state format."""

from __future__ import annotations

from collections.abc import Sequence

import torch

# V-JEPA 2-AC was trained on end-effector deltas and poses: xyz in metres,
# three rotation angles, then the gripper.
VJEPA_AC_DIM = 7
X, Y, Z, ROLL, PITCH, YAW, GRIPPER = range(VJEPA_AC_DIM)


class SlotMapping:
    """Places each rig dimension in one slot of a fixed-size vector, scaled, and leaves the rest at zero.

    Implements :class:`~vjepa_byor.interfaces.VJepaMapping`. For an XY stage
    measured in millimetres::

        SlotMapping(action_slots=[X, Y], state_slots=[X, Y], action_scale=1e-3, state_scale=1e-3)

    turns a ``[5, -5]`` mm move into ``[0.005, -0.005, 0, 0, 0, 0, 0]``.
    ``state_offset`` (in the state's own units, one per rig dimension) is
    subtracted before scaling, e.g. to put the middle of the travel at zero.
    """

    def __init__(
        self,
        action_slots: Sequence[int],
        state_slots: Sequence[int],
        *,
        action_scale: float | Sequence[float] = 1.0,
        state_scale: float | Sequence[float] = 1.0,
        state_offset: float | Sequence[float] = 0.0,
        dim: int = VJEPA_AC_DIM,
    ) -> None:
        self._dim = dim
        self._action_slots = self._check_slots(action_slots, "action")
        self._state_slots = self._check_slots(state_slots, "state")
        self._action_scale = self._per_slot(action_scale, len(self._action_slots), "action_scale")
        self._state_scale = self._per_slot(state_scale, len(self._state_slots), "state_scale")
        self._state_offset = self._per_slot(state_offset, len(self._state_slots), "state_offset")
        if (self._action_scale == 0).any():
            raise ValueError("action_scale must be non-zero, or distinct actions would map to the same input")

    def _check_slots(self, slots: Sequence[int], kind: str) -> list[int]:
        slots = list(slots)
        if not slots:
            raise ValueError(f"{kind}_slots must not be empty")
        if len(set(slots)) != len(slots):
            raise ValueError(f"{kind}_slots must not repeat a slot")
        if not all(0 <= s < self._dim for s in slots):
            raise ValueError(f"{kind}_slots must be between 0 and {self._dim - 1}")
        return slots

    @staticmethod
    def _per_slot(value: float | Sequence[float], n: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            return tensor.expand(n).clone()
        if tensor.shape != (n,):
            raise ValueError(f"{name} needs one value or {n} values, got {tensor.shape[0]}")
        return tensor

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def state_dim(self) -> int:
        return self._dim

    def _place(self, values: torch.Tensor, slots: list[int], scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != len(slots):
            raise ValueError(f"expected {len(slots)} values in the last dimension, got {values.shape[-1]}")
        dtype = values.dtype if values.is_floating_point() else torch.float32
        offset = offset.to(dtype=dtype, device=values.device)
        scale = scale.to(dtype=dtype, device=values.device)
        out = torch.zeros((*values.shape[:-1], self._dim), dtype=dtype, device=values.device)
        out[..., slots] = (values.to(dtype) - offset) * scale
        return out

    def map_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self._place(actions, self._action_slots, self._action_scale, torch.zeros(len(self._action_slots)))

    def map_states(self, states: torch.Tensor) -> torch.Tensor:
        return self._place(states, self._state_slots, self._state_scale, self._state_offset)
