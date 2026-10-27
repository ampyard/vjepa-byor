1. **Goal image generator**
   - Takes a photo and an English command and returns a goal image
   - Keeps the scene, camera angle and objects consistent; only the commanded change appears
   - Runs in under ~30 seconds per request

2. **Action vocabulary and adapter**
   - Defines a small fixed set of actions (e.g. 8 directions × 3 step sizes) as tensors, plus a rig state (current position)
   - Converts any action tensor to a hardware command string and back again
   - Maps rig actions and states to V-JEPA 2-AC's 7-D action/state format (e.g. an XY move becomes `[dx, dy, 0, 0, 0, 0, 0]`); the mapping is pluggable so a custom-sized input layer can replace it later
   - Generic parts live in the core library; our 8 × 3 vocabulary, XY state, mapping and codec live in the ESP32 rig plugin
   - Validated with unit tests and the core conformance tests (module 10), no hardware needed

3. **Fine-tuning data collector**
   - Records (before photo, rig state, action, after photo) samples automatically
   - Works with any registered rig plugin, selected by name (`--rig sim`, `--rig esp32-stepper`); no rig-specific code in the collector
   - Exports a clean dataset of at least a few hundred samples

4. **Fine-tuned V-JEPA predictor**
   - Fine-tunes only the V-JEPA 2-AC predictor; the encoder stays frozen and dataset latents are encoded once ahead of time
   - Fits in 8 GB VRAM (bf16; LoRA on the predictor if full fine-tuning does not fit)
   - Takes a current photo, rig state and action and outputs a predicted next-state embedding
   - On held-out data, predicts closer to the true outcome than both a "no change" baseline and the original (not fine-tuned) predictor
   - Inference fast enough to score a full action basket in a few seconds

5. **Scorer**
   - Returns a single error number between a predicted state and the goal state
   - Ranks a correct action above wrong ones on a labelled test set
   - Same score for identical inputs (deterministic)

6. **Planner loop**
   - Scores every action in the basket from the current photo and rig state, and picks the lowest error
   - Replans after each step and stops when error falls below a threshold
   - Works end to end in simulation before touching hardware

7. **ESP32 hardware controller**
   - Receives commands over WiFi or Bluetooth and drives the stepper motors
   - Acknowledges each command and reports the rig's position (state) when the move is complete
   - Has a safe stop and handles out-of-range commands
   - Firmware lives in the ESP32 rig plugin; a round-trip test keeps its wire format in sync with the Python codec

8. **Camera capture service**
   - Takes a photo on request with a fixed position and lighting
   - Returns the image in the same format and resolution as the fine-tuning data

9. **Live demo dashboard**
   - Shows the current photo, goal image, candidate scores and chosen action side by side
   - Lets the audience type a command and watch the loop run step by step
   - Shows a run log they can replay

10. **Core interfaces and plugin system**
    - Defines the protocols every rig implements (ActionSpace, RigState, VJepaMapping, CommandCodec, Transport, Camera) and a Rig bundle that groups them under one name
    - Discovers installed rig plugins automatically via Python entry points
    - Ships conformance tests and a fake in-memory transport so plugin authors can validate a rig without hardware
    - Proven by building our ESP32 rig and the simulated rig as plugins with no changes to the core

11. **Simulated rig plugin**
    - Implements the same interfaces as the real rig, including position reporting after each move
    - Produces photos in the same format and resolution as the camera service
    - Lets modules 3, 4 and 6 run end to end with no hardware attached

12. **PyPI packaging and release**
    - Package name `vjepa-byor` (import `vjepa_byor`) reserved on PyPI with a placeholder release
    - Core library installs on its own, with no hardware or POC dependencies; V-JEPA weights are not bundled
    - README covers loading V-JEPA 2-AC, writing a rig plugin (using ours as the example) and the collect → encode → finetune → eval workflow
    - CI runs the unit and conformance tests on every change

Suggested order: 10 first (everything else plugs into it), then 2 and 11, so modules 3, 4, 5 and 6 can be built and tested in simulation before the hardware (7, 8) is ready.

For the demo, modules 1, 5 and 9 work well as standalone showpieces even before the hardware is ready.
