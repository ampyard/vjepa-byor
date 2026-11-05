"""Smoke test with the real V-JEPA 2-AC weights; opt in with VJEPA_BYOR_REAL_MODEL=1 (needs CUDA and about 7 GB)."""

import os

import pytest
import torch

from vjepa_byor.actions import DiscreteActionSpace, SlotMapping
from vjepa_byor.actions.vjepa_mapping import X, Y
from vjepa_byor.predictor import ActionPredictor, default_checkpoint, load_pretrained

pytestmark = pytest.mark.skipif(
    os.environ.get("VJEPA_BYOR_REAL_MODEL") != "1" or not torch.cuda.is_available() or not default_checkpoint().is_file(),
    reason="set VJEPA_BYOR_REAL_MODEL=1, with CUDA and the weights in the torch.hub cache",
)


def test_scores_a_basket_with_the_real_model():
    model = load_pretrained()
    space = DiscreteActionSpace.compass([1, 5, 20])
    mapping = SlotMapping([X, Y], [X, Y], action_scale=1e-3, state_scale=1e-3)
    predictor = ActionPredictor(model.encoder.cuda(), model.predictor.cuda(), mapping)
    photo = torch.randint(0, 256, (3, 240, 320), dtype=torch.uint8)
    out = predictor.predict(predictor.encode(photo), torch.tensor([100.0, 75.0]), space.actions())
    assert out.shape == (24, 256, 1408) and torch.isfinite(out).all()
