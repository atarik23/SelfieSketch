# Hardware and purchasing

**Budget ceiling:** 150 BAM. Prices below are *planning allowances*, not live verified quotes. Confirm vendor availability, shipping and exact part specifications before ordering.

| Item | Qty | Allowance (BAM) |
|---|---:|---:|
| ESP32 DevKit | 1 | 20 |
| SG90 **positional** micro servos | 3 | 24 |
| PCA9685 PWM driver | 1 | 13 |
| Regulated external 5 V, 3 A power supply | 1 | 18 |
| Wires, connectors and prototyping board | 1 set | 10 |
| Craft sticks / thin plywood | 1 set | 5 |
| Rigid wooden base | 1 | 8 |
| Spring, clothespin, elastic pen holder | 1 set | 4 |
| Glue, screws, cable ties, spacers | 1 set | 8 |
| Pens and paper | 1 set | 6 |
| **Estimated total** | | **116** |

**Reserve:** 34 BAM for shipping, spare servo and unexpected materials.

## Mechanical concept
Two approximately 80 mm links connected with shoulder and elbow servos, plus a third servo for lifting the pen. Build from wooden sticks or thin plywood with a rigid base; avoid flexible cardboard as the primary structural material. The usable drawing region must be measured after assembly and calibrated; do not assume the theoretical 160 mm maximum reach is a rectangular workspace.

## Electrical safety
- Power the servos from a regulated external supply, **not** the ESP32 3.3 V pin.
- Connect the servo supply ground and ESP32 ground together.
- Verify 3.3 V I²C compatibility with the PCA9685 board and wiring.
- Ensure mechanical travel does not jam the servo horns.
- Provide a way to disconnect motor power quickly.
- Confirm positional SG90 models; continuous-rotation variants cannot provide direct angle positioning.

## Procurement checklist
- [ ] Confirm electronics vendors, costs and shipping
- [ ] Buy ESP32, three positional servos, PCA9685 and power supply
- [ ] Buy or reuse wires and assembly materials
- [ ] Obtain wood, mounting hardware, pen and paper
- [ ] Measure actual link lengths and usable joint ranges
