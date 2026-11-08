# vjepa-byor

**Bring Your Own Robot:** fine-tune V-JEPA 2-AC to predict what your robot's actions will do.

## What is a rig?

A **rig** is your robot setup: the actuators that move, the controller that drives them, and the
camera that watches the result. A desk-sized XY stage with stepper motors, an ESP32 and a webcam is a
rig, and so is a simulated one. "Robot" is the public-facing word; "rig" is the term used in the code
(`Rig`, `RigState`, `--rig`).

The library has no hardware code of its own. To use it with your robot, you describe your rig to it:
which actions it can take, what its position looks like, how to talk to it, and how to see it.

## Adding your rig

A rig plugin implements the protocols in `vjepa_byor.interfaces`, bundles them in a `vjepa_byor.Rig`,
and registers it under the `vjepa_byor.rigs` entry-point group:

```toml
[project.entry-points."vjepa_byor.rigs"]
my-rig = "my_rig_package:make_rig"
```

The rig's name must match the name it is registered under: `make_rig` above returns
`Rig(name="my-rig", ...)`, otherwise `load_rig("my-rig")` refuses it.

There are two ways to talk to the hardware:

- **Text commands** (serial, WiFi, Bluetooth): give the rig a `CommandCodec`, which turns an action into
  a command line and the reply back into a state, and a `Transport` that carries the lines.
- **Action tensors** (ROS 2, a vendor SDK, ...): give the rig an `ActionTransport` and no codec. Its
  `execute(action, timeout)` carries out the move and returns the state the rig reports.

A ROS 2 robot needs no plugin: the built-in `ros2` rig is described by a YAML file
(`--rig ros2 --rig-config robot.yaml`, see `vjepa_byor.ros2.config`). Install with
`pip install vjepa-byor[ros2]` in a venv created with `--system-site-packages` from a sourced ROS 2
environment, since `rclpy` is not on PyPI. Without `rclpy`, `available_rigs()` does not list `ros2`.

Validate it without hardware by subclassing `vjepa_byor.testing.conformance.RigConformanceTests`
(install with `pip install vjepa-byor[testing]`) and backing the rig with a
`vjepa_byor.testing.FakeTransport` (or `FakeActionTransport`) that answers like your device.

## Trying it without hardware

The library includes `sim`, a simulated XY stage with a rendered top-down camera, so collection,
fine-tuning and planning run end to end with nothing attached. The examples below use it; with your own
rig, use its registered name in place of `sim`.

```
pip install vjepa-byor
vjepa-byor collect --rig sim --out data/sim-try --num-samples 200 --settle-time 0
```

- **Actions:** 8 compass directions × 3 step sizes (1, 5 and 20 mm by default), as `[dx_mm, dy_mm]`
- **State:** `XYState(x_mm, y_mm)`, reported after every move; travel is 200 × 150 mm, starting in the middle
- **V-JEPA mapping:** millimetres to metres in the end effector's x and y slots: `[dx, dy, 0, 0, 0, 0, 0]`
- **Wire format:** the JSON-line codec over an in-memory transport; out-of-range moves are refused with
  `{"error": "out of range"}` and the carriage stays put
- **Camera:** 256 × 256 RGB `uint8` by default: an orange puck on a gridded mat, with a red and a blue
  landmark block. Anti-aliased, so a 1 mm move changes the photo; deterministic, so a position always
  gives the same photo

Options (`--rig-config sim.json`): `step_sizes_mm`, `travel_mm`, `start_mm`, `resolution` (`[height, width]`),
`reply_timeout`.

```python
from vjepa_byor import load_rig

with load_rig("sim") as rig:
    print(rig.execute(rig.action_space.by_label("NE 5mm")))   # XYState(x_mm=105.0, y_mm=80.0)
    photo = rig.camera.capture()                               # uint8 (3, 256, 256)
```

It is built only from the library's public building blocks (`vjepa_byor.sim`), so it doubles as an
example of a rig.

### The sim as a ROS 2 robot

`python -m vjepa_byor.sim.ros2_bridge` runs the sim as an ordinary ROS 2 node (needs a sourced ROS 2
environment and `vjepa-byor[ros2]`):

```bash
python -m vjepa_byor.sim.ros2_bridge --ros-args -p start_mm:="[40.0, 100.0]"
ros2 topic pub --once /rig/cmd geometry_msgs/msg/Vector3Stamped "{vector: {x: 0.02, y: -0.02}}"
ros2 topic echo /rig/state
```

| Name | Type | |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | `rgb8`, 256 × 256 |
| `/rig/state` | `sensor_msgs/JointState` | joints `x`, `y`, positions in metres |
| `/rig/cmd` (subscribed) | `geometry_msgs/Vector3Stamped` | relative move, `vector.x` / `vector.y` in metres |
| `/rig/reset` (service) | `std_srvs/Trigger` | carriage back to `start_mm` |

Photo and state share one `header.stamp` and go out at `rate_hz` (10) and after every command, including
a refused one, which leaves the position unchanged. Parameters: `travel_mm`, `start_mm`, `resolution`,
`rate_hz`, `camera_frame`, `stage_frame`.

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

## Fine-tuning the predictor

Only V-JEPA 2-AC's action-conditioned predictor is fine-tuned; the ViT-g encoder stays frozen. Install the
model code's dependencies with `pip install vjepa-byor[vjepa]`. The weights are not bundled (they have their
own license): download `vjepa2-ac-vitg.pt` (about 11 GB) from `https://dl.fbaipublicfiles.com/vjepa2/` to
`~/.cache/torch/hub/checkpoints/`, or pass `--checkpoint PATH`.

```
vjepa-byor encode   --data data/sim-run1                        # frozen encoder, once per frame
vjepa-byor finetune --data data/sim-run1 --out checkpoints/sim  # train split; best epoch on val is kept
vjepa-byor eval     --data data/sim-run1 --finetuned checkpoints/sim --split test
```

- **encode** writes `<dataset>/latents/latents.pt`: every frame's layer-normalised latents (256 tokens × 1408,
  bf16). A rerun after `collect --resume` encodes only the new frames.
- **finetune** trains on the latents, so the encoder is never in GPU memory. The objective is upstream's
  teacher-forced step: before latents + mapped action + mapped state → after latents, L1 loss. bf16 autocast,
  AdamW, warmup + cosine, activation checkpointing. `--lora-rank 16` trains LoRA adapters instead of every
  weight; `--max-vram-gb 8` caps GPU memory to check a run fits an 8 GB card. The rig's `VJepaMapping` comes
  from the rig the dataset was collected with (`--rig` / `--rig-config` to override); a mapping that is not
  7-D gets fresh, custom-sized action and state input layers.
- **eval** reports the mean L1 error of the fine-tuned predictor, the original one and a "no change" baseline
  (the before latents), over all samples and over those where the rig moved.

In code, `vjepa_byor.predictor.ActionPredictor(encoder, predictor, rig.mapping).predict_basket(photo, state,
rig.action_space)` predicts the next-state latents for every action in one batch.
