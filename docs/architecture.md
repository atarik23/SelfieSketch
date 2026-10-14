# Software architecture

```text
Webcam -> alignment validation -> portrait extraction -> vector polylines
    -> stroke ordering -> paper coordinate mapping -> path interpolation
    -> 2-link IK + joint-limit validation -> execution backend
                                         |-> 2D simulator (initial)
                                         |-> serial -> ESP32 -> PCA9685 -> 3 servos (future)
```

**Units:** millimeters for lengths and coordinates, radians for joint angles. Shoulder origin is (0, 0), with X to the right and Y upward in the planar model. Servo zero offsets and real drawing-paper coordinates will be calibrated later. The current ideal kinematic model does not yet account for mechanical backlash or the pen's offset.

**Hardware-independent contract (planned):** A drawing contains ordered polylines with explicit pen-up and pen-down transitions. Simulation and physical execution should consume the same validated trajectory.

**Current implementation:** analytical FK/IK, reachable-point checks, geometric paths, and an animated Matplotlib demo.

**Planned:** camera UI (PySide6), MediaPipe face alignment, OpenCV contours, stroke ordering, serial transport, ESP32 firmware, calibration and experiments.
