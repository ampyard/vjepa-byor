"""Fine-tunes only the V-JEPA 2-AC predictor, on latents encoded ahead of time.

The objective is upstream's teacher-forced step: from the before photo's
latents, the mapped action and the mapped state, predict the after photo's
latents, with an L1 loss on layer-normalised tokens. The encoder is never
loaded. Training runs in bf16 autocast with float32 master weights, AdamW and
a warmup + cosine learning rate; activation checkpointing is on by default.

``lora_rank > 0`` trains low-rank adapters instead of every weight (see
:mod:`vjepa_byor.predictor.lora`), for when full fine-tuning does not fit.
Either way the best epoch on the validation split is kept, merged and saved.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vjepa_byor.predictor.encoding import autocast
from vjepa_byor.predictor.inference import predict_next
from vjepa_byor.predictor.input_layers import adapt_input_layers
from vjepa_byor.predictor.latent_cache import LatentDataset
from vjepa_byor.predictor.loading import save_finetuned
from vjepa_byor.predictor.lora import apply_lora, merge_lora

DEFAULT_LR_FULL = 2e-5
DEFAULT_LR_LORA = 2e-4


@dataclass
class FinetuneConfig:
    epochs: int = 10
    batch_size: int = 8
    lr: float | None = None
    """Peak learning rate; defaults to 2e-5 for full fine-tuning and 2e-4 for LoRA."""
    weight_decay: float = 0.04
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    lora_rank: int = 0
    """0 trains every predictor weight; a positive rank trains LoRA adapters only."""
    lora_alpha: float | None = None
    activation_checkpointing: bool = True
    patience: int | None = None
    """Stop after this many epochs without a better validation loss."""
    seed: int = 0

    @property
    def learning_rate(self) -> float:
        if self.lr is not None:
            return self.lr
        return DEFAULT_LR_LORA if self.lora_rank > 0 else DEFAULT_LR_FULL


@dataclass
class EpochLog:
    epoch: int
    train_loss: float | None
    val_loss: float | None
    lr: float
    seconds: float


@dataclass
class FinetuneResult:
    directory: Path
    best_epoch: int
    best_val_loss: float | None
    trainable_params: int
    peak_vram_bytes: int | None
    history: list[EpochLog] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["directory"] = str(self.directory)
        return d


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float()).abs().mean()


def _batches(dataset: LatentDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _step_loss(predictor: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    z, target = batch["z_before"].to(device), batch["z_after"].to(device)
    with autocast(device):
        pred = predict_next(predictor, z, batch["action"].to(device), batch["state"].to(device))
    return l1_loss(pred, target)


@torch.no_grad()
def validation_loss(predictor: nn.Module, dataset: LatentDataset, batch_size: int, device: torch.device) -> float:
    was_training = predictor.training
    predictor.eval()
    total, count = 0.0, 0
    for batch in _batches(dataset, batch_size, shuffle=False, seed=0):
        n = len(batch["z_before"])
        total += float(_step_loss(predictor, batch, device)) * n
        count += n
    predictor.train(was_training)
    return total / count


def _param_groups(predictor: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for p in predictor.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def finetune(
    predictor: nn.Module,
    train: LatentDataset,
    val: LatentDataset | None,
    config: FinetuneConfig,
    out_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    info: dict[str, Any] | None = None,
    on_epoch: Callable[[EpochLog], None] | None = None,
) -> FinetuneResult:
    """Fine-tune ``predictor`` (the original V-JEPA 2-AC one, on the CPU) and save the best epoch to ``out_dir``.

    ``info`` is stored with the weights (dataset, rig, ...). The predictor is
    moved to ``device`` and left there, merged and in eval mode, with the best
    weights loaded. Epoch 0 in the history is the predictor before training.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)

    replaced = adapt_input_layers(predictor, action_dim=train.actions.shape[-1], state_dim=train.states.shape[-1])
    if config.lora_rank > 0:
        apply_lora(predictor, config.lora_rank, config.lora_alpha)
        for name in replaced:
            getattr(predictor, name).requires_grad_(True)
    else:
        predictor.requires_grad_(True)
    if hasattr(predictor, "use_activation_checkpointing"):
        predictor.use_activation_checkpointing = config.activation_checkpointing
    predictor.to(device)
    trainable = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.AdamW(
        _param_groups(predictor, config.weight_decay), lr=config.learning_rate, fused=device.type == "cuda"
    )
    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup = max(1, round(total_steps * config.warmup_fraction))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    def snapshot() -> dict[str, torch.Tensor]:
        return {k: v.detach().to("cpu", copy=True) for k, v in predictor.state_dict().items()}

    history: list[EpochLog] = []
    start = time.perf_counter()
    best_loss = validation_loss(predictor, val, config.batch_size, device) if val is not None else None
    best_epoch, best_state, stale = 0, snapshot(), 0
    history.append(EpochLog(0, None, best_loss, 0.0, time.perf_counter() - start))
    if on_epoch is not None:
        on_epoch(history[-1])

    for epoch in range(1, config.epochs + 1):
        predictor.train()
        start = time.perf_counter()
        total, count = 0.0, 0
        for batch in _batches(train, config.batch_size, shuffle=True, seed=config.seed + epoch):
            loss = _step_loss(predictor, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_([p for p in predictor.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            scheduler.step()
            n = len(batch["z_before"])
            total += float(loss.detach()) * n
            count += n
        val_loss = validation_loss(predictor, val, config.batch_size, device) if val is not None else None
        history.append(EpochLog(epoch, total / count, val_loss, scheduler.get_last_lr()[0], time.perf_counter() - start))
        if on_epoch is not None:
            on_epoch(history[-1])

        if val_loss is None or best_loss is None or val_loss < best_loss:
            best_loss, best_epoch, best_state, stale = val_loss, epoch, snapshot(), 0
        else:
            stale += 1
            if config.patience is not None and stale >= config.patience:
                break

    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    del optimizer
    predictor.load_state_dict(best_state)
    merge_lora(predictor)
    predictor.eval()

    result = FinetuneResult(Path(out_dir), best_epoch, best_loss, trainable, peak, history)
    save_finetuned(predictor, out_dir, {**(info or {}), "config": asdict(config), "result": result.to_dict()})
    return result
