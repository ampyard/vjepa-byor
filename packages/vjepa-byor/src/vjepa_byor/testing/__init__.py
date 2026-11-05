"""Tools for testing rig plugins without hardware.

The conformance suite lives in :mod:`vjepa_byor.testing.conformance` and needs
pytest (``pip install vjepa-byor[testing]``).
"""

from vjepa_byor.testing.fake_transport import FakeActionTransport, FakeTransport

__all__ = ["FakeActionTransport", "FakeTransport"]
