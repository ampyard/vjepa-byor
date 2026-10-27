import json

import pytest

from vjepa_byor import register_rig, unregister_rig
from vjepa_byor.cli import main
from vjepa_byor.data import read_samples


@pytest.fixture(autouse=True)
def line_rig_registered(marker_rig_factory):
    register_rig("line", marker_rig_factory)
    yield
    unregister_rig("line")


def collect(out, *extra):
    return main(["collect", "--rig", "line", "--out", str(out), "--settle-time", "0", *extra])


def test_collect_then_verify(tmp_path, capsys):
    assert collect(tmp_path / "ds", "--num-samples", "30", "--steps-per-episode", "10") == 0
    assert len(read_samples(tmp_path / "ds")) == 30
    assert "30/30 samples" in capsys.readouterr().out

    assert main(["collect", "--verify", str(tmp_path / "ds")]) == 0
    assert capsys.readouterr().out.startswith("OK: 30 samples in 3 episodes")


def test_verify_fails_on_a_broken_dataset(tmp_path, capsys):
    collect(tmp_path / "ds", "--num-samples", "10")
    (tmp_path / "ds/frames/ep000000/003.png").unlink()
    assert main(["collect", "--verify", str(tmp_path / "ds")]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_existing_dataset_needs_resume(tmp_path, capsys):
    collect(tmp_path / "ds", "--num-samples", "10")
    assert collect(tmp_path / "ds", "--num-samples", "20") == 2
    assert "--resume" in capsys.readouterr().err
    assert collect(tmp_path / "ds", "--num-samples", "20", "--resume") == 0
    assert len(read_samples(tmp_path / "ds")) == 20


def test_rig_config_is_passed_to_the_rig_factory(tmp_path):
    config = tmp_path / "rig.json"
    config.write_text(json.dumps({"start": 0}))
    assert collect(tmp_path / "ds", "--num-samples", "5", "--rig-config", str(config)) == 0
    first = read_samples(tmp_path / "ds")[0]
    assert first.state_before["x"] in (1, 5)  # sync move from 0: only +1 and +5 are in range


def test_state_bounds_and_reject_rate(tmp_path):
    assert collect(tmp_path / "ds", "--num-samples", "40", "--state-bounds", "0:20", "--reject-rate", "0") == 0


def test_bad_state_bounds(tmp_path, capsys):
    with pytest.raises(SystemExit):
        collect(tmp_path / "ds", "--state-bounds", "0-20")


def test_unknown_rig(tmp_path, capsys):
    assert main(["collect", "--rig", "nope", "--out", str(tmp_path)]) == 2
    assert "no rig named 'nope'" in capsys.readouterr().err


def test_rig_and_out_are_required(capsys):
    assert main(["collect"]) == 2
