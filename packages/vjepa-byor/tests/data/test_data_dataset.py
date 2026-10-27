import json
import shutil

import pytest
import torch
from PIL import Image

from vjepa_byor.data import CollectedDataset, CollectorConfig, CollectRun, collect, ensure_splits, export, verify
from vjepa_byor.data import dataset as dataset_module
from vjepa_byor.data.policy import EdgeAwareRandomWalk
from vjepa_byor.data.storage import MANIFEST, read_samples


def run(rig, root, *, resume=False, **config):
    config = CollectorConfig(**{"settle_time": 0, "steps_per_episode": 10, **config})
    policy = EdgeAwareRandomWalk(rig.action_space.actions(), reject_rate=0.2)
    collect(rig, root, CollectRun(config=config, resume=resume, policy=policy, handle_signals=False))


@pytest.fixture(scope="module")
def collected(tmp_path_factory, fake_sim_factory):
    path = tmp_path_factory.mktemp("collected") / "ds"
    run(fake_sim_factory(2, 2), path, num_samples=200)  # starts in a corner, so some moves get rejected
    return path


@pytest.fixture
def root(tmp_path, collected):
    """A fresh copy of a 200-sample, 20-episode dataset; tests may change it."""
    shutil.copytree(collected, tmp_path / "ds")
    return tmp_path / "ds"


def rewrite_manifest(root, edit):
    lines = (root / MANIFEST).read_text().splitlines()
    records = [json.loads(line) for line in lines]
    edit(records)
    (root / MANIFEST).write_text("".join(json.dumps(r) + "\n" for r in records))


# Splits


def test_splits_are_deterministic_and_disjoint(root, tmp_path, fake_sim_factory):
    splits = ensure_splits(root)
    assert set(splits) == {s.episode_id for s in read_samples(root)}
    assert list(splits.values()).count("train") == 16
    assert list(splits.values()).count("val") == list(splits.values()).count("test") == 2

    run(fake_sim_factory(2, 2), tmp_path / "again", num_samples=200)
    assert ensure_splits(tmp_path / "again") == splits

    parts = [CollectedDataset(root, name) for name in ("train", "val", "test")]
    episode_sets = [{s.episode_id for s in p.samples} for p in parts]
    assert sum(len(p) for p in parts) == 200
    for i in range(3):
        for j in range(i + 1, 3):
            assert not episode_sets[i] & episode_sets[j]


def test_seed_changes_the_split(tmp_path, root):
    other = tmp_path / "copy"
    shutil.copytree(root, other)
    assert ensure_splits(root, seed=0) != ensure_splits(other, seed=1)


def test_splits_survive_a_resume(root, fake_sim_factory):
    before = ensure_splits(root)
    run(fake_sim_factory(), root, num_samples=400, resume=True)
    after = ensure_splits(root)
    assert len(after) == 40
    assert {e: after[e] for e in before} == before
    assert json.loads((root / "splits.json").read_text())["episodes"] == after


def test_stored_split_settings_win(root):
    first = ensure_splits(root, seed=0)
    assert ensure_splits(root, seed=99, ratios=(0.5, 0.25, 0.25)) == first


# Reading


def test_items_hold_full_size_images_and_tensors(root):
    ds = CollectedDataset(root)
    item = ds[0]
    assert item["before"].dtype == torch.uint8 and item["before"].shape == (3, 48, 64)
    assert item["after"].shape == (3, 48, 64)
    sample = item["sample"]
    assert torch.equal(item["state_before"], torch.tensor(sample.state_before_tensor))
    assert torch.equal(item["action"], torch.tensor(sample.action))


def test_images_are_loaded_lazily(root, monkeypatch):
    calls = []
    real = dataset_module.load_frame
    monkeypatch.setattr(dataset_module, "load_frame", lambda path: calls.append(path) or real(path))
    ds = CollectedDataset(root)
    assert calls == []
    ds[3]
    assert len(calls) == 2


def test_transform_hook(root):
    def halve(item):
        item["before"] = item["before"][:, ::2, ::2]
        return item

    assert CollectedDataset(root, transform=halve)[0]["before"].shape == (3, 24, 32)


def test_filters(root):
    everything = CollectedDataset(root)
    ok = CollectedDataset(root, include_statuses=["ok"])
    moved = CollectedDataset(root, drop_no_change=True)
    assert 0 < len(ok) < len(everything)
    assert all(s.status == "ok" for s in ok.samples)
    assert all(s.changed for s in moved.samples) and len(moved) < len(everything)
    with pytest.raises(ValueError, match="unknown statuses"):
        CollectedDataset(root, include_statuses=["failed"])


# Export


def test_export_with_filters_is_a_valid_dataset(root, tmp_path):
    n = export(root, tmp_path / "out", split="val", drop_no_change=True)
    assert n == len(CollectedDataset(root, "val", drop_no_change=True))
    report = verify(tmp_path / "out")
    assert report.ok, report.format()
    exported_frames = {p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*.png")}
    referenced = {f for s in read_samples(tmp_path / "out") for f in (s.before_frame, s.after_frame)}
    assert exported_frames == referenced


def test_export_refuses_a_non_empty_directory(root, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        export(root, tmp_path / "out")


# Verify


def test_verify_detects_a_missing_frame(root):
    (root / "frames/ep000001/004.png").unlink()
    report = verify(root)
    assert not report.ok
    assert any("ep000001/004.png is missing" in e for e in report.errors)


def test_verify_detects_a_wrong_frame_size(root):
    Image.new("RGB", (10, 10)).save(root / "frames/ep000002/000.png")
    report = verify(root)
    assert any("is 10x10" in e for e in report.errors)


def test_verify_detects_a_broken_chain(root):
    def break_state(records):
        records[5]["state_before"] = {"x": 999}

    rewrite_manifest(root, break_state)
    report = verify(root)
    assert any("state after step 4 is not the state before step 5" in e for e in report.errors)


def test_verify_detects_a_frame_out_of_place(root):
    def swap(records):
        records[5]["before_frame"] = records[5]["after_frame"]

    rewrite_manifest(root, swap)
    errors = verify(root).errors
    assert any("should be" in e for e in errors)
    assert any("step 4 ends on" in e for e in errors)


def test_verify_detects_duplicate_ids(root):
    rewrite_manifest(root, lambda records: records.append(records[0]))
    errors = verify(root).errors
    assert any("appears 2 times" in e for e in errors)


def test_verify_warns_about_a_truncated_line(root):
    with open(root / MANIFEST, "ab") as f:
        f.write(b'{"sample')
    report = verify(root)
    assert report.ok and report.warnings


def test_verify_of_a_non_dataset(tmp_path):
    report = verify(tmp_path)
    assert not report.ok and "meta.json is missing" in report.errors[0]
