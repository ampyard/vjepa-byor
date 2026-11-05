"""Compares predictors on held-out data against a "no change" baseline.

For each sample the error is the mean L1 distance between the predicted and
the true after-photo latents. "No change" predicts the before photo's latents,
so a predictor that has learnt nothing about actions cannot beat it on the
samples where the rig moved. Errors are reported over all samples and over
only those where the rig moved (``changed``), which is where it matters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from vjepa_byor.predictor.encoding import autocast
from vjepa_byor.predictor.inference import predict_next
from vjepa_byor.predictor.latent_cache import LatentDataset

NO_CHANGE = "no change"


@dataclass
class EvalReport:
    samples: int
    changed: int
    errors: dict[str, dict[str, float]] = field(default_factory=dict)
    """Model name -> {"all": mean L1, "changed": mean L1 on the samples where the rig moved}."""

    def beats(self, name: str, other: str, subset: str = "changed") -> bool:
        return self.errors[name][subset] < self.errors[other][subset]

    def format(self) -> str:
        width = max(len(n) for n in self.errors)
        lines = [f"{self.samples} samples, {self.changed} where the rig moved", f"{'':{width}}  {'all':>8}  {'moved':>8}"]
        lines += [f"{n:{width}}  {e['all']:8.4f}  {e['changed']:8.4f}" for n, e in self.errors.items()]
        return "\n".join(lines)


@torch.no_grad()
def per_sample_errors(
    predictor: nn.Module | None, dataset: LatentDataset, *, batch_size: int = 16, device: torch.device | str | None = None
) -> torch.Tensor:
    """Mean L1 error of each sample; ``predictor=None`` is the "no change" baseline."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if predictor is not None:
        predictor.eval()
    out = []
    for batch in DataLoader(dataset, batch_size=batch_size):
        z, target = batch["z_before"].to(device), batch["z_after"].to(device)
        if predictor is None:
            pred = z
        else:
            with autocast(device):
                pred = predict_next(predictor, z, batch["action"].to(device), batch["state"].to(device))
        out.append((pred.float() - target).abs().mean(dim=(1, 2)).cpu())
    return torch.cat(out)


def evaluate(
    dataset: LatentDataset,
    predictors: Mapping[str, nn.Module],
    *,
    batch_size: int = 16,
    device: torch.device | str | None = None,
) -> EvalReport:
    """Errors of the "no change" baseline and of each predictor (already on ``device``) on ``dataset``."""
    changed = dataset.changed
    report = EvalReport(samples=len(dataset), changed=int(changed.sum()))
    models: dict[str, nn.Module | None] = {NO_CHANGE: None, **predictors}
    for name, predictor in models.items():
        errors = per_sample_errors(predictor, dataset, batch_size=batch_size, device=device)
        moved = errors[changed]
        report.errors[name] = {
            "all": float(errors.mean()),
            "changed": float(moved.mean()) if len(moved) else float("nan"),
        }
    return report
