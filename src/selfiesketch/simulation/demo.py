"""Run with: python -m selfiesketch.simulation.demo"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from selfiesketch.robotics.kinematics import PlanarArm
from selfiesketch.planning.shapes import square


def main():
    arm = PlanarArm()
    path = square(65, 45, 35)
    # Interpolate each edge so the animation traces continuous segments.
    samples = []
    for start, end in zip(path, path[1:]):
        for i in range(25):
            t = i / 25
            samples.append((start[0] * (1-t) + end[0] * t,
                            start[1] * (1-t) + end[1] * t))
    samples.append(path[-1])
    angles = [arm.inverse_kinematics(x, y) for x, y in samples]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-20, 180)
    ax.set_ylim(-20, 180)
    ax.grid(True, alpha=0.3)
    ax.set_title("SelfieSketch | hardware-free arm simulation")
    (links,) = ax.plot([], [], "o-", linewidth=3)
    (trace,) = ax.plot([], [], "-", linewidth=1.5)

    def update(frame):
        origin, elbow, tip = arm.joint_positions(*angles[frame])
        links.set_data([origin[0], elbow[0], tip[0]],
                       [origin[1], elbow[1], tip[1]])
        trace.set_data([p[0] for p in samples[:frame+1]],
                       [p[1] for p in samples[:frame+1]])
        return links, trace

    animation = FuncAnimation(fig, update, frames=len(samples),
                              interval=35, blit=True, repeat=False)
    plt.show()
    return animation


if __name__ == "__main__":
    main()
