"""End-to-end collection with the simulated rig (module 11).

TODO(module 11): this runs as soon as a rig named ``sim`` is installed; until
then it is skipped. Check it actually runs once the sim plugin lands.
"""

import pytest

from vjepa_byor import available_rigs
from vjepa_byor.cli import main
from vjepa_byor.data import verify


@pytest.mark.skipif("sim" not in available_rigs(), reason="needs the sim rig plugin (module 11)")
def test_sim_rig_collects_a_few_hundred_samples(tmp_path):
    out = tmp_path / "sim"
    assert main(["collect", "--rig", "sim", "--out", str(out), "--num-samples", "300", "--settle-time", "0"]) == 0
    report = verify(out)
    assert report.ok and report.samples == 300, report.format()
