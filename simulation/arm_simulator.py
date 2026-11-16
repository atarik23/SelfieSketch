import numpy as np
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle, Circle
from robotics.kinematics import DrawingArm

# ---------------- settings (match your robot) ----------------
L1 = 10.0                  # cm, stick 1 length
L2 = 10.0                  # cm, stick 2 length
S2_DIR = -1                # -1: elbow bends clockwise as servo 2 increases, +1: other way

PEN_UP_ANGLE = 90
PEN_DOWN_ANGLE = 30
PEN_CONTACT_ANGLE = 45     # pen touches paper when servo 3 <= this

PAPER = (-2.0, 7.0, 12.5, 9.5)    # x, y, width, height in cm
HOME = (90, 90, PEN_UP_ANGLE)
SPEED = 1.0                # animation speed (1 = real SG90 speed)


class ArmSimulator:
    def __init__(self):
        self.robot = DrawingArm(link1=L1, link2=L2, elbow_direction=S2_DIR)
        self.s = np.array(HOME, dtype=float)   # servo 1, 2, 3 angles
        self.ink_x, self.ink_y = [], []
        self._syncing = False
        self._build_window()
        self._redraw()

    # ================= Arduino-like API =================
    def send(self, s1, s2, s3):
        """Set all three servos at once (degrees 0-180)."""
        self._move_to(s1, s2, s3, animate=True)

    def write(self, servo, angle):
        """Set one servo (1, 2 or 3), like Arduino servo.write(angle)."""
        target = self.s.copy()
        target[servo - 1] = angle
        self._move_to(*target, animate=True)

    def delay(self, ms):
        """Wait, like Arduino delay(ms)."""
        plt.pause(ms / 1000.0)

    def pen_up(self):
        self.write(3, PEN_UP_ANGLE)

    def pen_down(self):
        self.write(3, PEN_DOWN_ANGLE)

    def home(self):
        self.send(*HOME)

    def move_to(self, x, y):
        """Move the pen to an XY position (cm), with pen lifted."""
        s1, s2 = self.robot.inverse(x, y)

        self.pen_up()
        self.send(s1, s2, PEN_UP_ANGLE)

    
    
    def line_to(self, x, y, step=0.1):
        """Draw a straight line using Cartesian interpolation."""
        _, start = self._pen_xy(self.s[0], self.s[1])

        distance = np.hypot(x - start[0], y - start[1])
        steps = max(1, int(np.ceil(distance / step)))

        targets = []

        for i in range(1, steps + 1):
            t = i / steps

            px = start[0] + t * (x - start[0])
            py = start[1] + t * (y - start[1])

            s1, s2 = self.robot.inverse(px, py)
            targets.append((s1, s2))

        self.pen_down()

        for s1, s2 in targets:
            self.send(s1, s2, PEN_DOWN_ANGLE)
            plt.pause(0.02)


    def clear(self):
        self.ink_x, self.ink_y = [], []
        self._redraw()

    # ================= simulation =================
    
    def _pen_xy(self, s1, s2):
        """Return elbow and pen positions for visualization."""
        t1 = np.radians(s1)

        elbow = np.array([
            self.robot.link1 * np.cos(t1),
            self.robot.link1 * np.sin(t1)
        ])

        pen = np.array(self.robot.forward(s1, s2))

        return elbow, pen


    def _ink(self):
        if self.s[2] <= PEN_CONTACT_ANGLE:
            _, pen = self._pen_xy(self.s[0], self.s[1])
            self.ink_x.append(pen[0])
            self.ink_y.append(pen[1])
        elif self.ink_x and not np.isnan(self.ink_x[-1]):
            self.ink_x.append(np.nan)   # lift pen -> break the line
            self.ink_y.append(np.nan)

    def _move_to(self, s1, s2, s3, animate):
        target = np.clip([s1, s2, s3], 0, 180)
        start = self.s.copy()
        steps = max(1, int(np.abs(target - start).max()))   # 1 degree per step
        dt = (0.1 / 60) / SPEED                              # SG90 ~0.1 s / 60 deg
        for i in range(1, steps + 1):
            self.s = start + (target - start) * i / steps
            self._ink()
            if animate and i % 2 == 0:
                self._redraw()
                plt.pause(max(2 * dt, 0.001))
        self.s = target
        self._ink()
        self._sync_sliders()
        self._redraw()

    # ================= window =================
    def _build_window(self):
        self.fig = plt.figure("2-DOF Drawing Arm Simulator", figsize=(11, 8))

        self.ax = self.fig.add_axes([0.05, 0.28, 0.62, 0.67])
        r = L1 + L2
        self.ax.set_xlim(-r - 1, r + 1)
        self.ax.set_ylim(-r * 0.35, r + 1)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Top view (cm)")
        px, py, pw, ph = PAPER
        self.ax.add_patch(Rectangle((px, py), pw, ph, facecolor="white",
                                    edgecolor="black", zorder=1))
        self.ax.add_patch(Rectangle((-2, -1.5), 4, 3, color="0.4", zorder=2))
        (self.ink_line,) = self.ax.plot([], [], color="navy", lw=1.6, zorder=3)
        (self.arm_line,) = self.ax.plot([], [], color="tab:orange", lw=6,
                                        solid_capstyle="round", zorder=4)
        (self.joints,) = self.ax.plot([], [], "o", color="black", ms=8, zorder=5)
        self.pen_dot = Circle((0, 0), 0.5, edgecolor="navy", zorder=6)
        self.ax.add_patch(self.pen_dot)

        # pen side view
        self.ax_pen = self.fig.add_axes([0.72, 0.62, 0.25, 0.30])
        self.ax_pen.set_xlim(-3, 3)
        self.ax_pen.set_ylim(-0.5, 4)
        self.ax_pen.set_xticks([])
        self.ax_pen.set_yticks([])
        self.ax_pen.set_title("Pen lift (side view)")
        self.ax_pen.axhline(0, color="black", lw=2)
        (self.pen_side,) = self.ax_pen.plot([], [], color="navy", lw=5,
                                            solid_capstyle="round")

        self.info = self.fig.text(0.72, 0.38, "", family="monospace", fontsize=10)

        sliders = []
        for i, name in enumerate(["Servo 1 (shoulder)", "Servo 2 (elbow)", "Servo 3 (pen)"]):
            ax = self.fig.add_axes([0.15, 0.18 - i * 0.05, 0.5, 0.03])
            sl = Slider(ax, name, 0, 180, valinit=self.s[i], valstep=1)
            sl.on_changed(self._on_slider)
            sliders.append(sl)
        self.sliders = sliders

        def button(x, y, label, fn):
            b = Button(self.fig.add_axes([x, y, 0.12, 0.05]), label)
            b.on_clicked(lambda e: fn())
            return b
        self.buttons = [
            button(0.72, 0.20, "Pen Up", self.pen_up),
            button(0.85, 0.20, "Pen Down", self.pen_down),
            button(0.72, 0.13, "Home", self.home),
            button(0.85, 0.13, "Clear paper", self.clear),
            button(0.72, 0.06, "Run program", lambda: my_program(self)),
        ]

    def _on_slider(self, _):
        if not self._syncing:
            self._move_to(*(sl.val for sl in self.sliders), animate=False)

    def _sync_sliders(self):
        self._syncing = True
        for sl, v in zip(self.sliders, self.s):
            sl.set_val(v)
        self._syncing = False

    def _redraw(self):
        s1, s2, s3 = self.s
        elbow, pen = self._pen_xy(s1, s2)
        down = s3 <= PEN_CONTACT_ANGLE
        self.arm_line.set_data([0, elbow[0], pen[0]], [0, elbow[1], pen[1]])
        self.joints.set_data([0, elbow[0]], [0, elbow[1]])
        self.pen_dot.center = pen
        self.pen_dot.set_facecolor("navy" if down else "white")
        self.ink_line.set_data(self.ink_x, self.ink_y)
        h = 0 if down else min(3.0, (s3 - PEN_CONTACT_ANGLE) / 15.0)
        self.pen_side.set_data([0, 0], [h, h + 1.0])
        self.info.set_text(f"Servo 1 : {s1:4.0f} deg\n"
                           f"Servo 2 : {s2:4.0f} deg\n"
                           f"Servo 3 : {s3:4.0f} deg\n\n"
                           f"Pen     : {'DOWN' if down else 'UP'}")
        self.fig.canvas.draw_idle()


# =====================================================================
# YOUR PROGRAM - runs when you press "Run program"
# Same calls you will later send to the Arduino.
# =====================================================================
"""def my_program(arm):
    arm.send(120, 80, PEN_UP_ANGLE)     # move above the paper
    arm.pen_down()
    arm.send(120, 100, PEN_DOWN_ANGLE)  # draw
    arm.write(1, 140)                   # move only servo 1
    arm.write(2, 80)                    # move only servo 2
    arm.send(120, 80, PEN_DOWN_ANGLE)
    arm.delay(300)                      # wait 300 ms
    arm.pen_up()
    arm.home()"""


def my_program(arm):
    arm.clear()

    # Move to the starting point
    arm.move_to(3, 10)

    # Draw a 4 x 4 cm square
    arm.line_to(7, 10)
    arm.line_to(7, 14)
    arm.line_to(3, 14)
    arm.line_to(3, 10)

    arm.pen_up()
    arm.home()


if __name__ == "__main__":
    arm = ArmSimulator()
    plt.show()
