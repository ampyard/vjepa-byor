import json
import os

import pytest
import torch

from vjepa_byor.data import DatasetFormatError, DatasetWriter, Sample, build_meta, load_frame, read_samples
from vjepa_byor.data.storage import (
    MANIFEST,
    manifest_is_truncated,
    next_episode_number,
    read_events,
    read_meta,
)


@pytest.fixture
def meta(marker_rig_factory):
    rig = marker_rig_factory()
    return build_meta(rig, rig.camera)


def random_image(height=32, width=48, seed=0):
    return torch.randint(0, 256, (3, height, width), dtype=torch.uint8, generator=torch.Generator().manual_seed(seed))


def make_sample(episode="ep000000", step=0, **overrides):
    fields = dict(
        sample_id=f"{episode}_s{step:04d}",
        episode_id=episode,
        step_index=step,
        before_frame=f"frames/{episode}/{step:03d}.png",
        after_frame=f"frames/{episode}/{step + 1:03d}.png",
        state_before={"x": 10},
        state_after={"x": 11},
        state_before_tensor=[10.0],
        state_after_tensor=[11.0],
        action_index=2,
        action_name="move +1",
        action=[1.0],
        command="MOVE 1",
        status="ok",
        changed=True,
        resolution=[32, 48],
        timestamp="2026-10-27T00:00:00.000+00:00",
    )
    return Sample(**{**fields, **overrides})


def test_round_trip(tmp_path, meta):
    images = [random_image(seed=i) for i in range(2)]
    with DatasetWriter.create(tmp_path, meta) as writer:
        assert writer.write_frame("ep000000", 0, images[0]) == "frames/ep000000/000.png"
        writer.write_frame("ep000000", 1, images[1])
        writer.append_sample(make_sample())
    assert read_meta(tmp_path) == meta
    assert read_samples(tmp_path) == [make_sample()]
    assert torch.equal(load_frame(tmp_path / "frames/ep000000/000.png"), images[0])
    assert torch.equal(load_frame(tmp_path / "frames/ep000000/001.png"), images[1])


def test_frames_are_lossless_png(tmp_path, meta):
    with DatasetWriter.create(tmp_path, meta) as writer:
        rel = writer.write_frame("ep000000", 0, random_image())
    assert (tmp_path / rel).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_frame_of_the_wrong_size_is_refused(tmp_path, meta):
    with DatasetWriter.create(tmp_path, meta) as writer, pytest.raises(ValueError, match="does not match"):
        writer.write_frame("ep000000", 0, random_image(height=10))


def test_sample_pointing_at_a_missing_frame_is_refused(tmp_path, meta):
    with DatasetWriter.create(tmp_path, meta) as writer:
        writer.write_frame("ep000000", 0, random_image())
        with pytest.raises(FileNotFoundError):
            writer.append_sample(make_sample())
    assert read_samples(tmp_path) == []


def test_frame_write_is_atomic(tmp_path, meta, monkeypatch):
    with DatasetWriter.create(tmp_path, meta) as writer:
        def crash(src, dst):
            raise OSError("power cut")

        monkeypatch.setattr(os, "replace", crash)
        with pytest.raises(OSError):
            writer.write_frame("ep000000", 0, random_image())
    assert not (tmp_path / "frames/ep000000/000.png").exists()


def test_truncated_last_line_is_skipped_and_dropped_on_resume(tmp_path, meta):
    with DatasetWriter.create(tmp_path, meta) as writer:
        writer.write_frame("ep000000", 0, random_image())
        writer.write_frame("ep000000", 1, random_image())
        writer.append_sample(make_sample())
    with open(tmp_path / MANIFEST, "ab") as f:
        f.write(b'{"sample_id": "ep000000_s0001", "epis')

    assert manifest_is_truncated(tmp_path)
    assert len(read_samples(tmp_path)) == 1

    DatasetWriter.resume(tmp_path, meta).close()
    assert not manifest_is_truncated(tmp_path)
    assert (tmp_path / MANIFEST).read_bytes().endswith(b"\n")
    assert len(read_samples(tmp_path)) == 1
    assert "truncated_line_dropped" in [e["event"] for e in read_events(tmp_path)]


def test_complete_json_without_newline_counts_as_truncated(tmp_path, meta):
    with DatasetWriter.create(tmp_path, meta):
        pass
    (tmp_path / MANIFEST).write_text(json.dumps(make_sample().to_dict()))
    assert manifest_is_truncated(tmp_path)
    assert read_samples(tmp_path) == []


def test_corrupt_line_in_the_middle_is_an_error(tmp_path, meta):
    with DatasetWriter.create(tmp_path, meta):
        pass
    good = json.dumps(make_sample().to_dict())
    (tmp_path / MANIFEST).write_text(f"{good}\nnot json\n{good}\n")
    with pytest.raises(DatasetFormatError, match="corrupt line"):
        read_samples(tmp_path)


def test_unknown_schema_version_is_refused(tmp_path, meta):
    DatasetWriter.create(tmp_path, meta).close()
    data = json.loads((tmp_path / "meta.json").read_text())
    data["schema_version"] = 2
    (tmp_path / "meta.json").write_text(json.dumps(data))
    with pytest.raises(DatasetFormatError, match="schema_version 2"):
        read_meta(tmp_path)


def test_create_refuses_an_existing_dataset(tmp_path, meta):
    DatasetWriter.create(tmp_path, meta).close()
    with pytest.raises(FileExistsError):
        DatasetWriter.create(tmp_path, meta)


def test_resume_refuses_a_different_vocabulary_or_resolution(tmp_path, meta):
    DatasetWriter.create(tmp_path, meta).close()
    other = type(meta).from_dict({**meta.to_dict(), "resolution": [64, 64], "action_labels": ["a"] * 4})
    with pytest.raises(DatasetFormatError, match="resolution.*vocabulary|vocabulary.*resolution"):
        DatasetWriter.resume(tmp_path, other)


def test_next_episode_skips_orphaned_frame_directories(tmp_path, meta):
    with DatasetWriter.create(tmp_path, meta) as writer:
        assert next_episode_number(tmp_path) == 0
        writer.write_frame("ep000000", 0, random_image())
        writer.write_frame("ep000000", 1, random_image())
        writer.append_sample(make_sample())
        assert next_episode_number(tmp_path) == 1
        writer.write_frame("ep000004", 0, random_image())  # crashed before any sample
    assert next_episode_number(tmp_path) == 5
