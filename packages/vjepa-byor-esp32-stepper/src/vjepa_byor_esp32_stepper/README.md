# vjepa-byor-esp32-stepper

[vjepa-byor](../vjepa-byor) rig plugin for the POC's XY stage: two stepper motors driven by an ESP32,
registered as `esp32-stepper`.

- **Actions:** 8 compass directions × 3 step sizes (1, 5 and 20 mm by default), as `[dx_mm, dy_mm]`
- **State:** `XYState(x_mm, y_mm)`, the carriage position the ESP32 reports after every move
- **V-JEPA mapping:** millimetres to metres in the end effector's x and y slots: `[dx, dy, 0, 0, 0, 0, 0]`
- **Wire format:** `MOVE <dx> <dy>` → `POS <x> <y>` or `ERR <message>`; see `codec.py`

Not here yet: the WiFi / Bluetooth transport and the firmware (backlog module 7) and the camera
(module 8). Until then, test with the firmware emulator:

```python
from vjepa_byor.testing import FakeTransport
from vjepa_byor_esp32_stepper import Esp32Emulator, Esp32StepperRig

with Esp32StepperRig(transport=FakeTransport(Esp32Emulator())) as rig:
    print(rig.execute(rig.action_space.by_label("NE 5mm")))   # XYState(x_mm=105.0, y_mm=80.0)
```
