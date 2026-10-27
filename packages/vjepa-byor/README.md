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

## Collecting a fine-tuning dataset

`vjepa-byor collect` drives any registered rig through short episodes of random actions and records every
transition as (before photo, rig state, action, after photo):

```
vjepa-byor collect --rig sim --out data/sim-run1 \
    --num-samples 1000 --steps-per-episode 20 --seed 0 --settle-time 0.5
vjepa-byor collect --rig sim --out data/sim-run1 --num-samples 2000 --resume   # adds 1000 more
vjepa-byor collect --verify data/sim-run1
```

`--num-samples` is the target size of the whole dataset, so a resumed run only collects what is missing.
`--rig-config rig.json` passes options to the rig's factory, and `--state-bounds 0:200,0:150` tells the
random walk the limits of the state tensor so it steers away from them (without bounds it learns from
rejections). Ctrl-C finishes the step in flight and stops cleanly; a second Ctrl-C aborts.

The rig needs a camera (`Rig.camera`). Since the interfaces have no "home" or "read position" command,
the collector learns where the rig is from its reply to a first, unrecorded move.

### Dataset layout

```
<dataset>/
├── meta.json         # schema_version, rig, action vocabulary, resolution [height, width], ...
├── manifest.jsonl    # one sample per line, append-only
├── events.jsonl      # rejections, failures, syncs, resumptions
├── splits.json       # episode -> train / val / test (created on first read)
└── frames/ep000000/000.png, 001.png, ...   # lossless PNGs exactly as captured
```

Step *k* of an episode has frame *k* as its before photo and frame *k+1* as its after photo, so photos are
stored once. The full format contract is in the docstring of `vjepa_byor.data.storage`.

Each manifest line is a sample:

| Field | Meaning |
|---|---|
| `sample_id`, `episode_id`, `step_index` | e.g. `ep000012_s0003`, `ep000012`, `3` |
| `before_frame`, `after_frame` | PNG paths relative to the dataset root |
| `state_before`, `state_after` | the rig's reported state, as `RigState.to_dict()` |
| `state_before_tensor`, `state_after_tensor` | the same, as `RigState.as_tensor()` |
| `action_index`, `action_name`, `action` | index into the vocabulary, its label, the action tensor |
| `command` | the wire line sent to the rig |
| `status` | `ok`, or `rejected` if the rig reported an error (the state is then unchanged) |
| `changed` | `state_after != state_before`, from the reported states |
| `resolution`, `timestamp`, `error` | `[height, width]`, ISO 8601 UTC, the rig's error message if rejected |

Read it back with `vjepa_byor.data.CollectedDataset(root, split="train", drop_no_change=False)`; splits are by
episode and never change once written. `vjepa_byor.data.export(root, out, split="train")` copies a split into
a new self-contained dataset.

Plugin authors can check the collector drives their rig by adding
`vjepa_byor.testing.collector_conformance.CollectorConformanceTests` next to `RigConformanceTests`; it needs
`rig.camera`, and `vjepa_byor.testing.fake_camera.MarkerCamera` can draw the emulated position.
