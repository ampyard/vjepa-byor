"""End-to-end collection with the built-in simulated rig (``sim``)."""

from vjepa_byor.cli import main
from vjepa_byor.data import verify


def test_sim_rig_collects_a_few_hundred_samples(tmp_path):
    out = tmp_path / "sim"
    assert main(["collect", "--rig", "sim", "--out", str(out), "--num-samples", "300", "--settle-time", "0"]) == 0
    report = verify(out)
    assert report.ok and report.samples == 300, report.format()
