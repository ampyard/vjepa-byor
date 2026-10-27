# vjepa-byor

**Bring Your Own Robot:** fine-tune V-JEPA 2-AC to predict what your robot's actions will do.

A rig plugin implements the protocols in `vjepa_byor.interfaces`, bundles them in a `vjepa_byor.Rig`,
and registers it under the `vjepa_byor.rigs` entry-point group:

```toml
[project.entry-points."vjepa_byor.rigs"]
my-rig = "my_rig_package:make_rig"
```

Validate it without hardware by subclassing `vjepa_byor.testing.conformance.RigConformanceTests`
(install with `pip install vjepa-byor[testing]`) and backing the rig with a
`vjepa_byor.testing.FakeTransport` that answers like your device.
