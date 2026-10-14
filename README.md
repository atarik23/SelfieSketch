# SelfieSketch

A low-cost, software-first robotic portrait drawing system for a three-person Robotics Fundamentals course project.

SelfieSketch aims to capture an aligned selfie, extract a simplified vector portrait, plan pen strokes, simulate a two-link planar drawing arm, and finally draw the portrait using a homemade BrachioGraph-inspired mechanism.

**Status:** initial software foundation; physical robot and camera pipeline not yet implemented.

## Scope

- Guided camera capture and facial landmark extraction (planned)
- Portrait-to-polyline conversion and stroke optimization (planned)
- 2-DOF planar forward/inverse kinematics (implemented)
- Hardware-free 2D plotting and animation (implemented)
- ESP32 with three positional servos (planned)

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -e ".[dev]"
python -m selfiesketch.simulation.demo
pytest
```

The demo shows a virtual two-link arm tracing a small square; no hardware is needed.

## Documentation

- [Project overview](docs/project-overview.md)
- [Architecture](docs/architecture.md)
- [Hardware and shopping list](docs/hardware.md)
- [Development roadmap](docs/roadmap.md)

## Background and attribution

Inspired by [BrachioGraph](https://www.brachiograph.art/) ([source](https://github.com/evildmp/BrachioGraph)). This project develops its own software stack and does not claim the original arm design as new. The robot is intended for supervised educational use.
