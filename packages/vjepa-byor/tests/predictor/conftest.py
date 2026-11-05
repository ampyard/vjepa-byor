"""Tiny stand-ins for V-JEPA 2-AC, and a small collected dataset to fine-tune them on.

``TinyEncoder`` and ``TinyPredictor`` take the same inputs and have the same
layer names as the upstream ViT-g encoder and AC predictor (``action_encoder``,
``state_encoder``, ``predictor_blocks[i].attn.qkv``, ...), so everything in
:mod:`vjepa_byor.predictor` runs on them on the CPU, without the weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from vjepa_byor import Rig
from vjepa_byor.actions import DiscreteActionSpace, JsonLineCodec, SlotMapping, VectorState
from vjepa_byor.data import CollectorConfig, CollectRun, collect
from vjepa_byor.testing import FakeTransport
from vjepa_byor.testing.fake_camera import MarkerCamera

DIM = 8  # latent width
TOKENS = 16  # 4 x 4 patches of the 256 x 256 crop
WIDTH, HEIGHT = 20, 20


class TinyEncoder(nn.Module):
    patch_size = 64

    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv3d(3, DIM, kernel_size=(2, 64, 64), stride=(2, 64, 64))
        self.calls = 0

    def forward(self, clip: torch.Tensor) -> torch.Tensor:  # (B, 3, 2, 256, 256) -> (B, 16, DIM)
        self.calls += 1
        return self.patch(clip).flatten(2).transpose(1, 2)


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(width, 3 * width)
        self.attn.proj = nn.Linear(width, width)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(width, 2 * width)
        self.mlp.fc2 = nn.Linear(2 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.attn.qkv(x).chunk(3, dim=-1)
        x = x + self.attn.proj(torch.softmax(q @ k.transpose(1, 2) / q.size(-1) ** 0.5, dim=-1) @ v)
        return x + self.mlp.fc2(torch.relu(self.mlp.fc1(x)))


class TinyPredictor(nn.Module):
    def __init__(self, width: int = 16, depth: int = 2, action_dim: int = 7) -> None:
        super().__init__()
        self.use_activation_checkpointing = False
        self.predictor_embed = nn.Linear(DIM, width)
        self.action_encoder = nn.Linear(action_dim, width)
        self.state_encoder = nn.Linear(action_dim, width)
        self.predictor_blocks = nn.ModuleList(TinyBlock(width) for _ in range(depth))
        self.predictor_norm = nn.LayerNorm(width)
        self.predictor_proj = nn.Linear(width, DIM)

    def forward(self, x: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.action_encoder(actions), self.state_encoder(states), self.predictor_embed(x)], dim=1)
        for block in self.predictor_blocks:
            tokens = block(tokens)
        return self.predictor_proj(self.predictor_norm(tokens[:, 2:]))


@pytest.fixture
def tiny_encoder() -> TinyEncoder:
    torch.manual_seed(0)
    return TinyEncoder()


@pytest.fixture
def tiny_predictor_factory():
    def make(**kwargs) -> TinyPredictor:
        torch.manual_seed(1)
        return TinyPredictor(**kwargs)

    return make


# A small XY rig, built from the core building blocks


@dataclass(frozen=True)
class GridState(VectorState):
    x: float
    y: float


class GridDevice:
    def __init__(self) -> None:
        self.x, self.y = WIDTH / 2, HEIGHT / 2
        self.codec = JsonLineCodec(GridState, 2)

    def __call__(self, line: str) -> str:
        dx, dy = self.codec.decode_action(line).tolist()
        if not (0 <= self.x + dx <= WIDTH and 0 <= self.y + dy <= HEIGHT):
            return '{"error": "out of range"}'
        self.x, self.y = self.x + dx, self.y + dy
        return self.codec.encode_state(GridState(self.x, self.y))


def make_grid_rig() -> Rig:
    device = GridDevice()
    return Rig(
        name="grid",
        action_space=DiscreteActionSpace.compass([1, 3]),
        state_type=GridState,
        mapping=SlotMapping([0, 1], [0, 1], action_scale=0.1, state_scale=0.1),
        codec=device.codec,
        transport=FakeTransport(device),
        camera=MarkerCamera(lambda: [device.x, device.y], [(0, WIDTH), (0, HEIGHT)], resolution=(64, 64), marker=8),
    )


@pytest.fixture
def grid_mapping() -> SlotMapping:
    return make_grid_rig().mapping


def collect_grid(root, num_samples: int, resume: bool = False):
    config = CollectorConfig(num_samples=num_samples, steps_per_episode=8, settle_time=0)
    return collect(make_grid_rig(), root, CollectRun(config=config, resume=resume, handle_signals=False))


@pytest.fixture(scope="session")
def grid_dataset_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("grid") / "ds"
    collect_grid(root, 160)
    return root


@pytest.fixture
def grid_dataset(grid_dataset_template, tmp_path):
    """A fresh copy of a 160-sample dataset (20 episodes) for each test."""
    import shutil

    root = tmp_path / "ds"
    shutil.copytree(grid_dataset_template, root)
    return root
