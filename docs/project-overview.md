# Project overview

**Course:** Robotics Fundamentals · **Team:** 3 · **Duration:** 6–8 weeks · **Hardware ceiling:** 150 BAM

## Concept
SelfieSketch is a low-cost robotic portrait-drawing system. A person aligns their face with an on-screen guide, captures a selfie, and previews a simplified line-art portrait. The software extracts vector strokes, optimizes their order, transforms paper coordinates into feasible arm joint angles, and executes them with a homemade two-link planar arm and pen-lift servo.

## Background
The design is inspired by [BrachioGraph](https://www.brachiograph.art/) ([GitHub](https://github.com/evildmp/BrachioGraph)), a low-cost pen plotter made from positional servos and household materials. Our project uses the same general mechanical concept but plans its own capture, sketch, kinematics, trajectory, simulation and control software.

## Main subsystems
1. **Guided camera capture:** webcam, face-shaped guide, detection, positioning and pose checks.
2. **Portrait generation:** landmarks and/or image edges produce recognizable line art.
3. **Stroke planning:** contours to polylines; simplification, scaling, pen-up travel optimization.
4. **Robot geometry:** 2-DOF planar FK/IK, limits, calibration and reachability.
5. **Simulation:** test trajectories on a virtual arm before hardware is available.
6. **Physical control:** ESP32, PCA9685 and three positional servos, with independent servo power.

## Success criteria
An aligned selfie is converted into a recognizable drawing and autonomously rendered on paper. The final report measures shape accuracy, drawing time, and effects of calibration and stroke ordering.

## Known challenges
Cheap servo backlash, link flex, pen contact pressure, workspace constraints, camera lighting and noisy contours. Start with squares and circles; do not attempt a full portrait before basic repeatability is demonstrated.

## Initial development strategy
Software-first: kinematics → simple paths → 2D simulation → guided capture → sketch generation → stroke planning → electronics and calibration → integration and evaluation.

## References
- https://www.brachiograph.art/
- https://github.com/evildmp/BrachioGraph
- https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
