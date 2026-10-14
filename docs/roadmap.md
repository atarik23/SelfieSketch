# Development roadmap

The project is software-first because physical parts may take time to arrive.

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Geometry and repository setup | FK/IK, tests, sample paths |
| 2 | Simulation and interpolation | Virtual arm drawing shapes |
| 3 | Guided camera interface | Face guide, alignment feedback |
| 4 | Sketch generation | Face landmarks/edges to vector strokes |
| 4–5 | Hardware build in parallel | Assembled, powered and calibrated arm |
| 6 | Integration | First physical portrait |
| 7 | Evaluation | Accuracy, speed and ordering comparisons |
| 8 | Final deliverables | Report, demo and reproducibility docs |

## Team workstreams
- Vision/UI: camera, alignment, landmarks, sketch extraction.
- Robotics/planning: kinematics, workspace, trajectory, simulator.
- Hardware/firmware: arm, electronics, serial protocol, calibration.

All three share integration and experimental evaluation.

## First sprint (no hardware required)
- [x] Initialize code and documentation
- [x] Implement basic 2-link FK/IK
- [x] Add reachable-point tests
- [x] Implement a simple 2D arm animation
- [ ] Add joint-space and Cartesian trajectory timing
- [ ] Add pen-up and pen-down transitions
- [ ] Design robot-independent stroke JSON schema
- [ ] Start face-guided camera UI
- [ ] Order hardware
