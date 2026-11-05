1. **Goal image generator**
   - Takes a photo and an English command and returns a goal image
   - Keeps the scene, camera angle and objects consistent; only the commanded change appears
   - Runs in under ~30 seconds per request

2. **Action vocabulary and adapter** ✅ Done
   - Defines a small fixed set of actions (e.g. 8 directions × 3 step sizes) as tensors, plus a rig state (current position)
   - Converts any action tensor to a hardware command string and back again
   - Maps rig actions and states to V-JEPA 2-AC's 7-D action/state format (e.g. an XY move becomes `[dx, dy, 0, 0, 0, 0, 0]`); the mapping is pluggable so a custom-sized input layer can replace it later
   - Generic parts live in the core library; our 8 × 3 vocabulary, XY state, mapping and codec live in the ESP32 rig plugin
   - Validated with unit tests and the core conformance tests (module 10), no hardware needed
   - ⏳ Not done yet: the ESP32 rig has no real transport or camera, so it runs only against its emulator (`Esp32Emulator`) until modules 7 and 8
   - ⏳ To change in module 7: the ESP32 rig's codec and emulator still use our own `MOVE`/`POS`/`ERR` protocol; they move to FluidNC's G-code and status reports

3. **Fine-tuning data collector** ✅ Done
   - Records (before photo, rig state, action, after photo) samples automatically
   - Works with any registered rig plugin, selected by name (`--rig sim`, `--rig esp32-stepper`); no rig-specific code in the collector
   - Exports a clean dataset of at least a few hundred samples
   - End-to-end collection with `--rig sim` passes (300 samples through the CLI, verified)
   - ⏳ Not done yet: a real-hardware collection run (needs modules 7 and 8)

4. **Fine-tuned V-JEPA predictor** ✅ Done
   - Fine-tunes only the V-JEPA 2-AC predictor; the encoder stays frozen and dataset latents are encoded once ahead of time
   - Fits in 8 GB VRAM (bf16; LoRA on the predictor if full fine-tuning does not fit)
   - Takes a current photo, rig state and action and outputs a predicted next-state embedding
   - On held-out data, predicts closer to the true outcome than both a "no change" baseline and the original (not fine-tuned) predictor
   - Inference fast enough to score a full action basket in a few seconds
   - Verified on a 2,000-sample `--rig sim` dataset (RTX 5060 Ti, capped at 8 GB): encoding took 56 s; full fine-tuning (10 epochs, about 12 min) peaked at 5.35 GiB, so LoRA (`--lora-rank`) is available but not needed. Test-split mean L1 where the rig moved: fine-tuned 0.180, no change 0.197, original 0.341. Scoring all 24 actions, including encoding the photo, takes about 0.25 s at 7.0 GiB peak
   - ⚠️ Risk for module 5: from one sample position, the true action's prediction was the closest of the 24 only 5 times. 20 mm moves mostly rank 1st–3rd; 1 mm and 5 mm moves are barely told apart (1 mm is about 1 px on the 256 crop, under one 16-px patch). Options: more data or epochs, a closer camera view in the sim and on the rig, or a scorer that weights the tokens that change
   - ⏳ Not done yet: fine-tuning on real-hardware data (needs modules 7 and 8)

5. **Scorer**
   - Returns a single error number between a predicted state and the goal state
   - Ranks a correct action above wrong ones on a labelled test set
   - Same score for identical inputs (deterministic)

6. **Planner loop**
   - Scores every action in the basket from the current photo and rig state, and picks the lowest error
   - Replans after each step and stops when error falls below a threshold
   - Works end to end in simulation before touching hardware

7. **ESP32 hardware controller (FluidNC)**
   - The ESP32 runs [FluidNC](http://wiki.fluidnc.com), open-source motion-control firmware, so we do not write or maintain our own
   - FluidNC's `config.yaml` (motor pins, steps per mm, speeds, 200 × 150 mm travel, homing, soft limits) is kept in the ESP32 rig plugin
   - Receives G-code over WiFi (telnet) or USB serial and drives the stepper motors
   - Each action is sent as a relative jog (`$J=G91 X<dx> Y<dy> F<feed>`). The plugin waits for `ok`, then polls the status report (`?`) until the state is `Idle` and reads the position from `MPos`
   - Homes on startup (`$H`, needs limit switches), so the reported position is absolute and the soft limits work
   - Refuses out-of-range moves: with soft limits on, a jog past the travel area is rejected with an `error:` reply and the carriage does not move, so no alarm needs clearing
   - Safe stop: jog cancel or feed hold (`!`), and soft reset (Ctrl-X) as the hard stop. `ALARM` states are raised as `RigError`
   - The plugin's codec changes from `MOVE`/`POS`/`ERR` to G-code jogs and FluidNC status reports. A FluidNC transport handles the multi-line exchange (`ok`, then status polling), so `Rig.execute` still sees one command and one reply
   - `Esp32Emulator` answers like FluidNC, so the unit and conformance tests still run without hardware; replies captured from the real board are stored as test fixtures so the codec is tested against real output

8. **Camera capture service**
   - Takes a photo on request with a fixed position and lighting
   - Returns the image in the same format and resolution as the fine-tuning data

9. **Live demo dashboard**
   - Shows the current photo, goal image, candidate scores and chosen action side by side
   - Lets the audience type a command and watch the loop run step by step
   - Shows a run log they can replay

10. **Core interfaces and plugin system** ✅ Done
    - Defines the protocols every rig implements (ActionSpace, RigState, VJepaMapping, CommandCodec, Transport, Camera) and a Rig bundle that groups them under one name
    - Discovers installed rig plugins automatically via Python entry points
    - Ships conformance tests and a fake in-memory transport so plugin authors can validate a rig without hardware
    - Proven by building our ESP32 rig and the simulated rig as plugins with no changes to the core interfaces; both pass the conformance and collector conformance tests

11. **Simulated rig plugin** ✅ Done
    - Implements the same interfaces as the real rig, including position reporting after each move
    - Produces photos in the same format and resolution as the camera service
    - Lets modules 3, 4 and 6 run end to end with no hardware attached
    - Same mechanics as the real stage (8 × 3 vocabulary, 200 × 150 mm travel, XY state in mm, same 7-D mapping), built only from core building blocks (`JsonLineCodec`, in-memory transport); renders an anti-aliased top-down view where every 1 mm move changes the photo
    - ⏳ Not done yet: module 8 does not exist, so the sim's 256 × 256 resolution (V-JEPA 2-AC's input size) is the target the camera service must match; modules 4 and 6 still need to be built and run against it

12. **PyPI packaging and release**
    - Package name `vjepa-byor` (import `vjepa_byor`) reserved on PyPI with a placeholder release
    - Core library installs on its own, with no hardware or POC dependencies; V-JEPA weights are not bundled
    - README covers loading V-JEPA 2-AC, writing a rig plugin (using ours as the example) and the collect → encode → finetune → eval workflow
    - CI runs the unit and conformance tests on every change

13. **ROS 2 support through a config file**
    - Any ROS 2 robot can be used as a rig without writing a plugin package: `vjepa-byor collect --rig ros2 --rig-config my_robot.yaml`
    - Ships inside `vjepa-byor` as the built-in `ros2` rig with an optional extra (`pip install vjepa-byor[ros2]`). ROS is imported only when `--rig ros2` is selected, so the core still installs and runs without ROS. `rclpy` comes from the user's sourced ROS 2 environment, not from PyPI, and the README says so
    - The config file describes:
      - the command interface: a topic or an action, its message type, and how a rig action fills the message
      - the state source: a topic, its message type, and which fields make up the rig state
      - the camera topic and the output resolution (256 × 256 by default, same as the fine-tuning data)
      - the action vocabulary and the V-JEPA mapping slots (reusing `SlotMapping`)
    - Supports common message types out of the box (`geometry_msgs/Twist`, `geometry_msgs/PoseStamped`, `sensor_msgs/JointState`, `control_msgs/FollowJointTrajectory`, `sensor_msgs/Image` and `CompressedImage`). For anything else, the config can name a custom converter function
    - A move counts as finished when the ROS action completes, or when the state stops changing within a set tolerance. The photo and the state are matched by message timestamps
    - Imports recorded rosbags as a `DataSource`, so existing robot logs become (before photo, rig state, action, after photo) samples
    - ✅ Core interface change done: besides the line `Transport` + `CommandCodec` pair, a rig can use an `ActionTransport` that takes action tensors and returns rig states directly (`execute(action, timeout)`), so ROS messages don't go through strings. `FakeActionTransport` and the conformance suite support it (codec tests are skipped), and the collector records `command: null` for such rigs. Existing plugins work unchanged
    - Validated by running the conformance tests and a `collect` run against the sim rig exposed through ROS 2 by a small bridge node, and against one Gazebo robot. CI runs the ROS tests in a ROS 2 container, separate from the main job
    - Stretch goal: expose the planner (module 6) as a ROS 2 action server, so a ROS stack can ask the robot to reach a goal image
    - Our own rig does not use ROS: the ESP32 with FluidNC (module 7) talks to the plugin directly

Suggested order: 10 first (everything else plugs into it), then 2 and 11, so modules 3, 4, 5 and 6 can be built and tested in simulation before the hardware (7, 8) is ready. Module 13 comes after the demo works and before or alongside the PyPI release (12).

For the demo, modules 1, 5 and 9 work well as standalone showpieces even before the hardware is ready.
