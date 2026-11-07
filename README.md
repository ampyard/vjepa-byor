# V-JEPA Physical AI POC

Type a command in plain English and a low-cost rig carries it out. An image-generation model imagines the goal state; a fine-tuned V-JEPA 2-AC predictor scores every candidate action against that goal; the best action is sent to an ESP32 running [FluidNC](http://wiki.fluidnc.com) that drives stepper motors. A new photo is taken and the loop repeats until the goal is reached.

> **What is a rig?** A rig is the physical robot setup: the actuators that move (here, stepper motors), the controller that drives them (here, an ESP32 running FluidNC), and the camera that watches the result. In this POC it is a small XY stage. "Robot" is the public-facing word; "rig" is the term used in the code and the docs (`Rig`, `RigState`, `--rig`). A simulated rig stands in for the hardware during development.

![Pipeline](vjepa_physical_ai_pipeline.png)

The modules and their acceptance criteria are in [backlog.md](backlog.md).

## Repository layout

This is a monorepo. One part is a library published to PyPI; the rest is specific to this POC.

```
vjepa-physical-ai-poc/
├── pyproject.toml                   # uv workspace root (not published)
├── backlog.md
│
├── packages/
│   ├── vjepa-byor/                  # core library, published to PyPI (backlog modules 2–5, 10, 12, 13)
│   ├── vjepa-byor-esp32-stepper/    # our rig (ESP32 + FluidNC), built as a plugin (rig-specific parts of module 2 + module 7)
│   └── vjepa-byor-sim/              # simulated rig, also a plugin (module 11)
│
├── apps/                            # POC-specific, never published
│   ├── goal_generator/              # module 1
│   ├── planner/                     # module 6 demo runner (planning loop on a chosen rig)
│   ├── camera_service/              # module 8
│   └── dashboard/                   # module 9
│
├── docker/                          # ROS 2 Jazzy + GPU container (see docker/README.md)
├── configs/                         # actions, rig and fine-tuning settings
├── data/                            # gitignored datasets
├── checkpoints/                     # gitignored fine-tuned weights
└── docs/
```

### Core library: `packages/vjepa-byor/`

**vjepa-byor — Bring Your Own Robot:** fine-tune V-JEPA 2-AC to predict what your robot's actions will do.

It fine-tunes your pretrained V-JEPA on your own robot's (before, action, after) data, and includes the tools to define that action set and collect that data. Install with `pip install vjepa-byor`; import as `vjepa_byor`.

Each robot it works with is described as a [rig](#v-jepa-physical-ai-poc) (actuators + controller + camera), supplied as a plugin.

```
src/vjepa_byor/
├── interfaces.py          # Protocols: ActionSpace, RigState, VJepaMapping, CommandCodec,
│                          #   Transport (text lines), ActionTransport (action tensors), Camera, DataSource
├── rig.py                 # Rig bundle: groups a rig's pieces under one registered name
├── registry.py            # plugin discovery via Python entry points
├── actions/               # module 2 (generic parts)
│   ├── vocabulary.py      #   discrete action sets as tensors (e.g. 8 compass directions × step sizes)
│   ├── state.py           #   rig state schema (VectorState: a few named numbers)
│   ├── vjepa_mapping.py   #   rig action/state → V-JEPA 2-AC 7-D action/state format (SlotMapping)
│   └── codecs/            #   reusable wire formats: action tensor ⇄ command line (JSON line, …)
├── data/                  # module 3
│   ├── sample.py          #   (before photo, rig state, action, after photo)
│   ├── collector.py       #   runs any DataSource and records samples
│   ├── sources/           #   generic sources built on a Rig + Camera
│   ├── storage.py         #   on-disk format and manifest
│   └── dataset.py         #   torch Dataset, splits, export
├── predictor/             # module 4
│   ├── loading.py         #   loads V-JEPA 2-AC (pinned torch.hub commit) and fine-tuned predictors
│   ├── encoding.py        #   photo → latents: resize/crop 256, ImageNet norm, layer norm
│   ├── latent_cache.py    #   encodes the dataset once with the frozen encoder
│   ├── input_layers.py    #   optional custom-sized action/state input layers
│   ├── finetune.py        #   predictor-only fine-tuning
│   ├── lora.py            #   fallback if full fine-tuning does not fit in VRAM
│   ├── evaluate.py        #   vs "no change" baseline and vs the original predictor
│   └── inference.py       #   predicts the outcome of a whole action basket in one batch
├── planning/              # module 5
│   ├── scorer.py          #   one error per candidate action vs the goal latents (L1, change-weighted)
│   └── ranking.py         #   where the true action ranks on a labelled split (vjepa-byor rank)
├── testing/               # conformance tests and a fake in-memory transport for plugin authors
├── ros2/                  # module 13: built-in `ros2` rig configured from a file; optional extra
│                          #   vjepa-byor[ros2], imported only with --rig ros2
└── cli.py                 # vjepa-byor collect | encode | finetune | eval | rank
```

### Rig plugins: `packages/vjepa-byor-<rig>/`

Every rig, including ours, is a plugin. A plugin provides:

| Piece | Implements | Required? |
|---|---|---|
| Action vocabulary | `ActionSpace` | Yes |
| Rig state | `RigState` | Yes |
| V-JEPA mapping | `VJepaMapping` | Yes, unless an existing mapping fits |
| Command codec | `CommandCodec` | Only for rigs driven with text command lines, unless a built-in one fits |
| Transport | `Transport` (text lines, paired with the codec) or `ActionTransport` (action tensors, no codec) | Yes, unless a built-in one fits |
| Camera | `Camera` | Only if a generic webcam does not work |
| Device firmware or its config | — | When the hardware needs it; prefer off-the-shelf firmware (we use FluidNC) |

```
vjepa-byor-esp32-stepper/
├── pyproject.toml         # registers the rig via entry points
├── src/vjepa_byor_esp32_stepper/
│   ├── actions.py         # 8 directions × 3 step sizes, XY state (mm)
│   ├── mapping.py         # XY move → [dx, dy, 0, 0, 0, 0, 0] (metres)
│   ├── codec.py           # G-code jog → FluidNC status report (MPos) | error / ALARM (module 7)
│   ├── emulator.py        # Python stand-in for FluidNC, for tests
│   ├── rig.py             # Esp32StepperRig factory; options come from --rig-config
│   └── transport.py       # WiFi (telnet) / USB serial; waits for ok, polls status until Idle (module 7)
├── fluidnc/config.yaml    # FluidNC machine config: pins, steps/mm, travel, homing, soft limits
└── tests/                 # runs the core library's conformance tests
```

A plugin registers itself in its `pyproject.toml`:

```toml
[project.entry-points."vjepa_byor.rigs"]
esp32-stepper = "vjepa_byor_esp32_stepper:Esp32StepperRig"
```

and is selected by name: `vjepa-byor collect --rig esp32-stepper`, or `--rig sim`.

### Simulated rig: `packages/vjepa-byor-sim/`

Registered as `sim`. It has the same mechanics as our real stage and is built only from the core library's
building blocks (the JSON-line codec and an in-memory transport), with a rendered top-down camera.

```
vjepa-byor-sim/
├── pyproject.toml
├── src/vjepa_byor_sim/
│   ├── stage.py           # XY carriage in mm; answers JSON-line commands, refuses out-of-range moves
│   ├── camera.py          # renders the mat, landmark blocks and carriage, 256 × 256 by default
│   └── rig.py             # SimRig factory; options come from --rig-config
└── tests/                 # conformance, collector conformance and rendering tests
```

## Running with ROS 2

On Ubuntu 24.04, ROS 2 Jazzy can be installed natively. Everywhere else (other Linux versions, Windows via WSL 2, macOS without a GPU), use the Docker container with ROS 2 Jazzy and the workspace: build it with `docker build -f docker/Dockerfile -t vjepa-ros2:jazzy .` and start a shell with `docker/run.sh`. Which setup to use, prerequisites, options and known issues are in [docker/README.md](docker/README.md).

## Design decisions

- **Dependencies point one way.** Plugins and apps import the core library; the core never imports them. The core contains no hardware code.
- **ROS 2 is optional and comes from a config file.** `--rig ros2 --rig-config robot.yaml` lets any ROS 2 robot be used as a rig without writing a plugin (backlog module 13). It ships in the core library as an extra, `vjepa-byor[ros2]`, and is imported only when selected. ROS is a way to talk to a robot, not hardware code, so this fits the rule above. `rclpy` comes from the user's ROS environment, not PyPI.
- **Two ways to talk to a rig.** Devices that speak text (serial, WiFi, Bluetooth, FluidNC's G-code) use a `CommandCodec` plus a line `Transport`. Rigs driven through typed messages (ROS 2, vendor SDKs) use an `ActionTransport`, which takes the action tensor and returns the rig state, with no codec and no strings in between. Samples from such rigs have no `command` line.
- **Our rig is a plugin, not a built-in.** It uses only the public interfaces. If the core has to change to make our rig work, a third party would have hit the same problem, so we fix the interface rather than add a special case.
- **The simulator is a plugin too.** The planner switches between simulation and hardware with `--rig`, with no code changes (backlog module 6).
- **Off-the-shelf firmware, not our own.** The ESP32 runs FluidNC, which already does homing, soft limits, position reporting and safe stops. We keep its `config.yaml` with the plugin. The codec is tested against replies captured from the real board, and the emulator answers like FluidNC.
- **Our own rig does not use ROS.** The plugin talks G-code to FluidNC directly; ROS support is for other people's robots.
- **We fine-tune; we do not train.** The V-JEPA 2-AC encoder stays frozen. Only the predictor is fine-tuned, on latents encoded once ahead of time, so the encoder is not in GPU memory during fine-tuning.
- **Rig actions are mapped to V-JEPA 2-AC's 7-D format first.** For example, an XY move becomes `[dx, dy, 0, 0, 0, 0, 0]`, which reuses the pretrained action layers and needs less data. Custom-sized input layers (`input_layers.py`) are the fallback for rigs that do not map naturally.
- **Every sample records the rig state.** The predictor needs the current position along with each action, so the hardware reports its position after every move and the collector stores it.
- **V-JEPA weights are not bundled.** Users load them from Meta's sources into their own setup; the weights have their own license.
- **Built for 8 GB GPUs.** bf16 throughout, with LoRA on the predictor as the fallback if full fine-tuning does not fit.

## Hardware baseline

V-JEPA 2-AC (ViT-g encoder, 1,012M params; AC predictor, 305M params) has been verified on an RTX 4060 Laptop GPU (8 GB) with torch 2.11 + CUDA 12.8:

- Scoring 64 candidate actions in one batch takes about 0.9 s.
- Peak VRAM during inference is 5.4 GB.

Fine-tuning the predictor (RTX 5060 Ti, capped at 8 GB with `--max-vram-gb 8`, batch 8, activation checkpointing):
full fine-tuning of all 305M predictor weights peaks at 5.35 GiB and takes about 70 s per epoch on 1,600 samples.

## Open decisions

- Dataset format; the current plan is image folders plus a JSONL manifest
- License for this repository
- Which ROS 2 distributions module 13 supports (current LTS first)
