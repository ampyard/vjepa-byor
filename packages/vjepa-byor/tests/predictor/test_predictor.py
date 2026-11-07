import json

import pytest
import torch
from torch import nn

from vjepa_byor.data import load_frame, read_samples
from vjepa_byor.predictor import (
    NO_CHANGE,
    ActionPredictor,
    FinetuneConfig,
    LatentDataset,
    LoRALinear,
    adapt_input_layers,
    apply_lora,
    encode_dataset,
    encode_frames,
    evaluate,
    finetune,
    load_cache,
    load_finetuned,
    merge_lora,
    per_sample_errors,
    predict_next,
    preprocess,
    read_finetuned_info,
)
from vjepa_byor.predictor.encoding import IMAGENET_MEAN

from .conftest import DIM, TOKENS, GridState, collect_grid, make_grid_rig

# Preprocessing and encoding


def test_preprocess_resizes_the_short_side_and_centre_crops():
    images = torch.zeros((2, 3, 120, 160), dtype=torch.uint8)
    assert preprocess(images).shape == (2, 3, 256, 256)


def test_preprocess_normalises_with_imagenet_statistics():
    mean = torch.tensor([round(m * 255) for m in IMAGENET_MEAN], dtype=torch.uint8)
    images = mean.view(1, 3, 1, 1).expand(1, 3, 256, 256).contiguous()
    assert preprocess(images).abs().max() < 0.01


def test_preprocess_rejects_float_images():
    with pytest.raises(ValueError, match="uint8"):
        preprocess(torch.zeros((1, 3, 256, 256)))


def test_encode_frames_is_layer_normalised_and_batch_independent(tiny_encoder):
    images = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)
    one_by_one = encode_frames(tiny_encoder, images, batch_size=1)
    together = encode_frames(tiny_encoder, images, batch_size=5)
    assert together.shape == (5, TOKENS, DIM)
    assert torch.allclose(one_by_one, together, atol=1e-5)
    assert torch.allclose(together.mean(-1), torch.zeros(5, TOKENS), atol=1e-5)


# Latent cache


def test_encodes_every_frame_once(grid_dataset, tiny_encoder):
    cache = encode_dataset(grid_dataset, tiny_encoder, "tiny", batch_size=7)
    samples = read_samples(grid_dataset)
    frames = {f for s in samples for f in (s.before_frame, s.after_frame)}
    assert sorted(cache.frames) == sorted(frames)
    assert len(frames) == len(samples) + 20  # n steps share n + 1 frames per episode
    assert cache.latents.shape == (len(frames), TOKENS, DIM) and cache.latents.dtype == torch.bfloat16


def test_cached_latents_match_encoding_the_photo(grid_dataset, tiny_encoder):
    cache = encode_dataset(grid_dataset, tiny_encoder, "tiny")
    s = read_samples(grid_dataset)[37]
    expected = encode_frames(tiny_encoder, load_frame(grid_dataset / s.after_frame).unsqueeze(0))[0]
    assert torch.allclose(cache.get(s.after_frame).float(), expected, atol=0.05)


def test_rerun_encodes_nothing_and_resume_only_new_frames(grid_dataset, tiny_encoder):
    encode_dataset(grid_dataset, tiny_encoder, "tiny", batch_size=1)
    first = tiny_encoder.calls
    encode_dataset(grid_dataset, tiny_encoder, "tiny", batch_size=1)
    assert tiny_encoder.calls == first

    before = load_cache(grid_dataset)
    collect_grid(grid_dataset, 176, resume=True)  # two more episodes: 16 steps, 18 frames
    cache = encode_dataset(grid_dataset, tiny_encoder, "tiny", batch_size=1)
    assert tiny_encoder.calls - first == 18
    assert cache.frames[: len(before.frames)] == before.frames
    assert torch.equal(cache.latents[: len(before.frames)], before.latents)


def test_another_model_id_starts_afresh(grid_dataset, tiny_encoder):
    encode_dataset(grid_dataset, tiny_encoder, "tiny", batch_size=1000)
    first = tiny_encoder.calls
    assert encode_dataset(grid_dataset, tiny_encoder, "other", batch_size=1000).model == "other"
    assert tiny_encoder.calls == first + 1


def test_latent_dataset_items(grid_dataset, tiny_encoder, grid_mapping):
    encode_dataset(grid_dataset, tiny_encoder, "tiny")
    data = LatentDataset(grid_dataset, grid_mapping)
    item = data[3]
    s = data.samples[3]
    assert item["z_before"].shape == item["z_after"].shape == (TOKENS, DIM)
    assert item["action"].shape == item["state"].shape == (1, 7)
    assert item["action"][0].tolist()[:2] == pytest.approx([v * 0.1 for v in s.action])
    assert item["state"][0].tolist()[:2] == pytest.approx([v * 0.1 for v in s.state_before_tensor])
    assert bool(item["changed"]) == s.changed


def test_latent_dataset_splits_are_disjoint(grid_dataset, tiny_encoder, grid_mapping):
    encode_dataset(grid_dataset, tiny_encoder, "tiny")
    ids = {name: {s.sample_id for s in LatentDataset(grid_dataset, grid_mapping, name).samples} for name in ("train", "val", "test")}
    assert ids["train"] and ids["val"] and ids["test"]
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert sum(map(len, ids.values())) == 160


def test_latent_dataset_needs_the_cache(grid_dataset, grid_mapping):
    with pytest.raises(FileNotFoundError, match="vjepa-byor encode"):
        LatentDataset(grid_dataset, grid_mapping)


def test_latent_dataset_spots_a_stale_cache(grid_dataset, tiny_encoder, grid_mapping):
    encode_dataset(grid_dataset, tiny_encoder, "tiny")
    collect_grid(grid_dataset, 168, resume=True)
    with pytest.raises(ValueError, match="missing from the latent cache"):
        LatentDataset(grid_dataset, grid_mapping)


# Prediction step and inference


def test_predict_next_is_layer_normalised(tiny_predictor_factory):
    out = predict_next(tiny_predictor_factory(), torch.randn(3, TOKENS, DIM), torch.randn(3, 1, 7), torch.randn(3, 1, 7))
    assert out.shape == (3, TOKENS, DIM)
    assert torch.allclose(out.mean(-1), torch.zeros(3, TOKENS), atol=1e-5)


def test_predict_basket_scores_every_action_in_chunks(tiny_encoder, tiny_predictor_factory):
    rig = make_grid_rig()
    model = ActionPredictor(tiny_encoder, tiny_predictor_factory(), rig.mapping, batch_size=5)
    photo = rig.camera.capture()
    state = GridState(10, 10)
    basket = model.predict_basket(photo, state, rig.action_space)
    assert basket.shape == (16, TOKENS, DIM)

    z = model.encode(photo)
    for i in (0, 7, 15):
        single = model.predict(z, state.as_tensor(), rig.action_space.actions()[i : i + 1])[0]
        assert torch.allclose(single, basket[i], atol=1e-5)
    assert not torch.allclose(basket[0], basket[15])


# Input layers


def test_input_layers_kept_for_7d_and_replaced_otherwise(tiny_predictor_factory):
    predictor = tiny_predictor_factory()
    original = predictor.action_encoder
    assert adapt_input_layers(predictor, 7, 7) == []
    assert predictor.action_encoder is original
    assert adapt_input_layers(predictor, 2, 3) == ["action_encoder", "state_encoder"]
    assert predictor.action_encoder.in_features == 2 and predictor.state_encoder.in_features == 3


# LoRA


def test_lora_starts_as_the_original_and_trains_only_adapters(tiny_predictor_factory):
    predictor = tiny_predictor_factory()
    x, a, s = torch.randn(2, TOKENS, DIM), torch.randn(2, 1, 7), torch.randn(2, 1, 7)
    before = predictor(x, a, s)
    assert apply_lora(predictor, rank=4) == 8  # 2 blocks x 4 layers
    assert torch.allclose(predictor(x, a, s), before, atol=1e-6)
    trainable = {n for n, p in predictor.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_a", "lora_b")) for n in trainable)


def test_merged_lora_gives_the_same_outputs(tiny_predictor_factory):
    predictor = tiny_predictor_factory()
    apply_lora(predictor, rank=4, alpha=8)
    for m in predictor.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_b)
    x, a, s = torch.randn(2, TOKENS, DIM), torch.randn(2, 1, 7), torch.randn(2, 1, 7)
    adapted = predictor(x, a, s)
    assert merge_lora(predictor) == 8
    assert not any(isinstance(m, LoRALinear) for m in predictor.modules())
    assert torch.allclose(predictor(x, a, s), adapted, atol=1e-5)
    assert set(predictor.state_dict()) == set(tiny_predictor_factory().state_dict())


# Fine-tuning


@pytest.fixture
def encoded(grid_dataset, tiny_encoder, grid_mapping):
    cache = encode_dataset(grid_dataset, tiny_encoder, "tiny")
    return {split: LatentDataset(grid_dataset, grid_mapping, split, cache=cache) for split in ("train", "val", "test")}


@pytest.mark.parametrize("lora_rank", [0, 4])
def test_finetune_improves_and_saves_the_best_epoch(encoded, tiny_predictor_factory, tmp_path, lora_rank):
    config = FinetuneConfig(epochs=6, batch_size=16, lr=3e-3, lora_rank=lora_rank, seed=0)
    result = finetune(tiny_predictor_factory(), encoded["train"], encoded["val"], config, tmp_path / "ft", device="cpu")
    losses = [e.val_loss for e in result.history]
    assert len(result.history) == 7 and result.history[0].train_loss is None
    assert result.best_val_loss == min(losses) < losses[0]
    assert result.history[-1].train_loss < result.history[1].train_loss

    total = sum(p.numel() for p in tiny_predictor_factory().parameters())
    assert (result.trainable_params == total) == (lora_rank == 0)

    loaded = load_finetuned(tmp_path / "ft", tiny_predictor_factory())
    assert per_sample_errors(loaded, encoded["val"], device="cpu").mean().item() == pytest.approx(result.best_val_loss, rel=1e-4)
    info = read_finetuned_info(tmp_path / "ft")
    assert info["config"]["lora_rank"] == lora_rank and info["result"]["best_epoch"] == result.best_epoch


def test_finetune_with_custom_sized_input_layers(grid_dataset, tiny_encoder, tiny_predictor_factory, tmp_path):
    from vjepa_byor.actions import SlotMapping

    mapping = SlotMapping([0, 1], [0, 1], action_scale=0.1, state_scale=0.1, dim=2)
    cache = encode_dataset(grid_dataset, tiny_encoder, "tiny")
    train, val = (LatentDataset(grid_dataset, mapping, split, cache=cache) for split in ("train", "val"))
    config = FinetuneConfig(epochs=1, batch_size=32, lr=1e-3, lora_rank=2)
    finetune(tiny_predictor_factory(), train, val, config, tmp_path / "ft", device="cpu")
    loaded = load_finetuned(tmp_path / "ft", tiny_predictor_factory())
    assert loaded.action_encoder.in_features == 2 and loaded.state_encoder.in_features == 2


def test_patience_stops_early(encoded, tiny_predictor_factory, tmp_path):
    config = FinetuneConfig(epochs=20, batch_size=16, lr=10.0, patience=2, grad_clip=0)  # diverges
    result = finetune(tiny_predictor_factory(), encoded["train"], encoded["val"], config, tmp_path / "ft", device="cpu")
    assert len(result.history) < 21


# Evaluation


def test_no_change_baseline_compares_before_with_after(encoded):
    data = encoded["test"]
    errors = per_sample_errors(None, data, device="cpu")
    expected = [(data[i]["z_before"] - data[i]["z_after"]).abs().mean().item() for i in range(len(data))]
    assert errors.tolist() == pytest.approx(expected, rel=1e-5)
    assert all(e == 0 for e, c in zip(errors.tolist(), data.changed.tolist()) if not c)


def test_evaluate_reports_every_model(encoded, tiny_predictor_factory):
    report = evaluate(encoded["test"], {"tiny": tiny_predictor_factory()}, device="cpu")
    assert list(report.errors) == [NO_CHANGE, "tiny"]
    assert report.samples == len(encoded["test"]) and 0 < report.changed <= report.samples
    assert report.beats(NO_CHANGE, "tiny")  # an untrained predictor is worse than predicting no change
    assert "moved" in report.format()


# CLI, with the tiny models standing in for the real ones


@pytest.fixture
def fake_models(monkeypatch, tiny_encoder, tiny_predictor_factory):
    from vjepa_byor import register_rig, unregister_rig
    from vjepa_byor.predictor import VJepaAC, loading

    def load_pretrained(checkpoint=None, *, encoder=True, predictor=True):
        return VJepaAC(tiny_encoder if encoder else None, tiny_predictor_factory() if predictor else None)

    monkeypatch.setattr(loading, "load_pretrained", load_pretrained)
    monkeypatch.setattr(loading, "build_architecture", lambda: (tiny_encoder, tiny_predictor_factory()))
    register_rig("grid", make_grid_rig)
    yield
    unregister_rig("grid")


def test_cli_encode_finetune_eval(grid_dataset, fake_models, tmp_path, capsys):
    from vjepa_byor.cli import main

    data = str(grid_dataset)
    assert main(["encode", "--data", data, "--device", "cpu"]) == 0
    assert "latent cache: 180 frames" in capsys.readouterr().out

    out = str(tmp_path / "ft")
    argv = ["finetune", "--data", data, "--out", out, "--device", "cpu", "--epochs", "3", "--lr", "3e-3"]
    assert main(argv) == 0
    assert "best epoch" in capsys.readouterr().out
    assert read_finetuned_info(out)["rig"] == "grid"

    report = tmp_path / "report.json"
    assert main(["eval", "--data", data, "--finetuned", out, "--device", "cpu", "--json", str(report)]) == 0
    printed = capsys.readouterr().out
    assert "fine-tuned beats both" in printed
    assert list(json.loads(report.read_text())["errors"]) == [NO_CHANGE, "original", "fine-tuned"]

    ranks = tmp_path / "ranks.json"
    argv = ["rank", "--data", data, "--finetuned", out, "--device", "cpu", "--scorer", "l1", "--scorer", "change-weighted"]
    assert main([*argv, "--json", str(ranks)]) == 0
    printed = capsys.readouterr().out
    assert "16 candidate actions" in printed and "ranked 1st" in printed
    assert [r["scorer"] for r in json.loads(ranks.read_text())] == ["l1", "change-weighted"]


def test_cli_finetune_needs_encoded_data(grid_dataset, fake_models, tmp_path):
    from vjepa_byor.cli import main

    with pytest.raises(FileNotFoundError, match="vjepa-byor encode"):
        main(["finetune", "--data", str(grid_dataset), "--out", str(tmp_path / "ft"), "--device", "cpu"])


def test_missing_weights_explain_where_to_get_them(tmp_path):
    from vjepa_byor.predictor import load_pretrained

    with pytest.raises(FileNotFoundError, match="dl.fbaipublicfiles.com"):
        load_pretrained(tmp_path / "nope.pt")

