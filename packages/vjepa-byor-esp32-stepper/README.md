# vjepa-byor-esp32-stepper

[vjepa-byor](../vjepa-byor) rig plugin for the POC's XY stage: two stepper motors driven by an ESP32
running [FluidNC](http://wiki.fluidnc.com), registered as `esp32-stepper`.

- **Actions:** 8 compass directions × 3 step sizes (1, 5 and 20 mm by default), as `[dx_mm, dy_mm]`
- **State:** `XYState(x_mm, y_mm)`, the carriage position FluidNC reports after every move
- **V-JEPA mapping:** millimetres to metres in the end effector's x and y slots: `[dx, dy, 0, 0, 0, 0, 0]`
- **Wire format (current):** `MOVE <dx> <dy>` → `POS <x> <y>` or `ERR <message>`; see `codec.py`
- **Wire format (module 7):** G-code jogs (`$J=G91 X<dx> Y<dy> F<feed>`) → `ok`, then the status report
  (`<Idle|MPos:x,y,z|...>`) once the move ends, or `error:` / `ALARM:` if refused

Not here yet: the FluidNC transport, G-code codec and `config.yaml` (backlog module 7) and the camera
(module 8). Until then, test with the emulator:

```python
from vjepa_byor.testing import FakeTransport
from vjepa_byor_esp32_stepper import Esp32Emulator, Esp32StepperRig

with Esp32StepperRig(transport=FakeTransport(Esp32Emulator())) as rig:
    print(rig.execute(rig.action_space.by_label("NE 5mm")))   # XYState(x_mm=105.0, y_mm=80.0)
```
