# vjepa-byor-sim

[vjepa-byor](../vjepa-byor) rig plugin for a simulated XY stage, registered as `sim`. It has the same
mechanics as the real stage ([esp32-stepper](../vjepa-byor-esp32-stepper)), so collection, fine-tuning
and planning run end to end with no hardware attached. Switch to the real stage with `--rig`.

- **Actions:** 8 compass directions × 3 step sizes (1, 5 and 20 mm by default), as `[dx_mm, dy_mm]`
- **State:** `XYState(x_mm, y_mm)`, reported after every move; travel is 200 × 150 mm, starting in the middle
- **V-JEPA mapping:** millimetres to metres in the end effector's x and y slots: `[dx, dy, 0, 0, 0, 0, 0]`
- **Wire format:** the core library's JSON-line codec, over an in-memory transport; out-of-range moves are
  refused with `{"error": "out of range"}` and the carriage stays put
- **Camera:** a rendered top-down view, 256 × 256 RGB `uint8` by default: an orange puck on a gridded mat,
  with a red and a blue landmark block. Anti-aliased, so a 1 mm move changes the photo; deterministic,
  so a position always gives the same photo

It is built only from the core library's building blocks, with no rig-specific codec or transport.

```
vjepa-byor collect --rig sim --out data/sim-run1 --num-samples 1000 --settle-time 0
```

Options (`--rig-config sim.json`): `step_sizes_mm`, `travel_mm`, `start_mm`, `resolution` (`[height, width]`),
`reply_timeout`.

```python
from vjepa_byor import load_rig

with load_rig("sim") as rig:
    print(rig.execute(rig.action_space.by_label("NE 5mm")))   # XYState(x_mm=105.0, y_mm=80.0)
    photo = rig.camera.capture()                               # uint8 (3, 256, 256)
```

## As a ROS 2 robot

`vjepa_byor_sim.ros2_bridge` runs the sim as an ordinary ROS 2 node (needs `rclpy`, e.g. in the
[ROS 2 container](../../docker/README.md)):

```bash
docker/run.sh python -m vjepa_byor_sim.ros2_bridge --ros-args -p start_mm:="[40.0, 100.0]"
docker/run.sh ros2 topic pub --once /rig/cmd geometry_msgs/msg/Vector3Stamped "{vector: {x: 0.02, y: -0.02}}"
docker/run.sh ros2 topic echo /rig/state
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
