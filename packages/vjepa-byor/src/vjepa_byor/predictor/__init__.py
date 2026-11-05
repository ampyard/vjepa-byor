"""Fine-tuning V-JEPA 2-AC's predictor on a collected dataset, evaluating it, and using it to score actions.

Workflow: :func:`encode_dataset` (frozen encoder, once) -> :func:`finetune`
(predictor only) -> :func:`evaluate` (vs "no change" and the original
predictor) -> :class:`ActionPredictor` (score an action basket). Loading the
real model needs ``pip install vjepa-byor[vjepa]`` and the weights; see
:mod:`vjepa_byor.predictor.loading`.
"""

from vjepa_byor.predictor.encoding import CROP_SIZE, encode_frames, preprocess
from vjepa_byor.predictor.evaluate import NO_CHANGE, EvalReport, evaluate, per_sample_errors
from vjepa_byor.predictor.finetune import EpochLog, FinetuneConfig, FinetuneResult, finetune
from vjepa_byor.predictor.inference import ActionPredictor, predict_next
from vjepa_byor.predictor.input_layers import adapt_input_layers
from vjepa_byor.predictor.latent_cache import LatentCache, LatentDataset, encode_dataset, load_cache
from vjepa_byor.predictor.loading import (
    MODEL_ID,
    VJepaAC,
    default_checkpoint,
    load_finetuned,
    load_pretrained,
    read_finetuned_info,
    save_finetuned,
)
from vjepa_byor.predictor.lora import LoRALinear, apply_lora, merge_lora

__all__ = [
    "CROP_SIZE",
    "MODEL_ID",
    "NO_CHANGE",
    "ActionPredictor",
    "EpochLog",
    "EvalReport",
    "FinetuneConfig",
    "FinetuneResult",
    "LatentCache",
    "LatentDataset",
    "LoRALinear",
    "VJepaAC",
    "adapt_input_layers",
    "apply_lora",
    "default_checkpoint",
    "encode_dataset",
    "encode_frames",
    "evaluate",
    "finetune",
    "load_cache",
    "load_finetuned",
    "load_pretrained",
    "merge_lora",
    "per_sample_errors",
    "predict_next",
    "preprocess",
    "read_finetuned_info",
    "save_finetuned",
]
