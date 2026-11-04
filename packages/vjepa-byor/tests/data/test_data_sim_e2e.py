"""End-to-end collection with the simulated rig plugin (``vjepa-byor-sim``).

The core does not depend on its plugins, so this skips when the core is
installed on its own; in the workspace the sim plugin is installed and it runs.
"""

import pytest

from vjepa_byor import available_rigs
from vjepa_byor.cli import main
from vjepa_byor.data import verify


@pytest.mark.skipif("sim" not in available_rigs(), reason="needs the sim rig plugin (vjepa-byor-sim)")
def test_sim_rig_collects_a_few_hundred_samples(tmp_path):
    out = tmp_path / "sim"
    assert main(["collect", "--rig", "sim", "--out", str(out), "--num-samples", "300", "--settle-time", "0"]) == 0
    report = verify(out)
    assert report.ok and report.samples == 300, report.format()
