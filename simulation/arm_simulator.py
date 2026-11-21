import json
import threading
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle, Circle
from robotics.kinematics import DrawingArm
from planning.shapes import interpolate_line

# Never pull the window to the front on its own.
plt.rcParams["figure.raise_window"] = False

# ---------------- settings (match your robot) ----------------
L1 = 10.0                  # cm, stick 1 length
L2 = 10.0                  # cm, stick 2 length
S2_DIR = -1                # -1: elbow bends clockwise as servo 2 increases, +1: other way

PEN_UP_ANGLE = 60
PEN_DOWN_ANGLE = 30
PEN_CONTACT_ANGLE = 45     # pen touches paper when servo 3 <= this

PAPER = (-2.0, 7.0, 12.5, 9.5)    # x, y, width, height in cm
HOME = (90, 90, PEN_UP_ANGLE)
SPEED = 0.5                # time scale for EVERYTHING: 1 = real time, 2 = twice as fast, 0.5 = half speed
SERVO_SPEED = 600.0        # deg/s, shoulder/elbow (SG90 is about 0.1 s per 60 deg)
PEN_SPEED = 150.0          # deg/s, pen servo - lift/lower gently (60 deg -> 0.4 s)
PEN_SETTLE = 0.15          # s to wait after the pen goes up or down (lets it stop bouncing)
POINT_DELAY = 0.02         # s (real time) to wait at each 0.1 cm step while drawing
FPS = 30                   # screen refreshes per second


class StopProgram(Exception):
    """Raised inside the worker thread when the window is closed."""


class ArmSimulator:
    """
    Threading model (this is what keeps the window responsive):

    * The robot program (my_program, Home, Pen Up, ...) runs in a background
      worker thread. It only changes numbers (servo angles, ink points) and
      sleeps with time.sleep(), exactly like code on an Arduino would.
    * The GUI thread never runs the program. A timer redraws the window
      FPS times per second from the current numbers.

    Matplotlib/Qt objects are only touched from the GUI thread; the worker
    only touches plain data guarded by self._lock.
    """

    def __init__(self):
        self.robot = DrawingArm(link1=L1, link2=L2, elbow_direction=S2_DIR)
        self.s = np.array(HOME, dtype=float)   # servo 1, 2, 3 angles
        self.ink_x, self.ink_y = [], []
        self._lock = threading.Lock()
        self._worker = None
        self._stop = False
        self._clock = time.perf_counter()     # schedule: when the arm should be done
        self._dirty = True
        self._syncing = False
        self._build_window()
        self._render()

        self._timer = self.fig.canvas.new_timer(interval=int(1000 / FPS))
        self._timer.add_callback(self._on_timer)
        self._timer.start()
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    # ================= Arduino-like API =================
    # (call these from my_program; they run in the worker thread)

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
        self._sleep(ms / 1000.0)

    def pen_up(self):
        self._move_pen(PEN_UP_ANGLE)

    def pen_down(self):
        self._move_pen(PEN_DOWN_ANGLE)

    def _move_pen(self, angle):
        if self.s[2] == angle:           # already there (line_to calls pen_down a lot)
            return
        self.write(3, angle)
        self._sleep(PEN_SETTLE)

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

        points = interpolate_line(start, (x, y), step)

        # Validate the entire path before drawing
        targets = [
            self.robot.inverse(px, py)
            for px, py in points[1:]
        ]

        self.pen_down()

        for s1, s2 in targets:
            self.send(s1, s2, PEN_DOWN_ANGLE)
            self._sleep(POINT_DELAY)

    def draw_path(self, points, step=0.1):
        """Draw a continuous path through a sequence of XY points."""
        if len(points) < 2:
            raise ValueError("A path must contain at least two points.")

        self.move_to(*points[0])

        for x, y in points[1:]:
            self.line_to(x, y, step)

        self.pen_up()

    def draw_paths(self, paths, step=0.1):
        """Draw multiple independent paths."""
        for path in paths:
            self.draw_path(path, step)

    def clear(self):
        with self._lock:
            self.ink_x, self.ink_y = [], []
            self._dirty = True

    # ================= simulation (data only, no GUI calls) =================

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
        # caller holds self._lock
        if self.s[2] <= PEN_CONTACT_ANGLE:
            _, pen = self._pen_xy(self.s[0], self.s[1])
            self.ink_x.append(pen[0])
            self.ink_y.append(pen[1])
        elif self.ink_x and not np.isnan(self.ink_x[-1]):
            self.ink_x.append(np.nan)   # lift pen -> break the line
            self.ink_y.append(np.nan)

    def _sleep(self, seconds):
        """Wait `seconds` of robot time.

        Many tiny time.sleep() calls each overshoot (especially on Windows),
        so the overshoot, not SPEED/POINT_DELAY, decided how long a drawing
        took. Instead keep a schedule: every wait moves the schedule forward,
        and we only sleep when we are ahead of it. Total time = sum of the
        requested waits, whatever the OS timer resolution is.

        `seconds` is real-robot time; SPEED scales ALL waits (servo motion,
        POINT_DELAY and delay()), so SPEED = 2 really is twice as fast.
        """
        if self._stop:
            raise StopProgram
        if seconds <= 0:
            return
        now = time.perf_counter()
        if now - self._clock > 0.1:      # fell behind / was idle: don't rush to catch up
            self._clock = now
        self._clock += seconds / SPEED
        ahead = self._clock - now
        if ahead > 0.002:
            time.sleep(ahead)

    def _move_to(self, s1, s2, s3, animate):
        target = np.clip(np.array([s1, s2, s3], dtype=float), 0, 180)
        start = self.s.copy()
        delta = np.abs(target - start)
        span = float(delta.max())                            # biggest turn, degrees
        steps = max(1, int(np.ceil(span)))                   # ~1 degree per animation step
        speeds = np.array([SERVO_SPEED, SERVO_SPEED, PEN_SPEED])
        duration = float((delta / speeds).max())             # slowest servo decides
        step_time = duration / steps                         # real-robot seconds per step
        for i in range(1, steps + 1):
            with self._lock:
                self.s = start + (target - start) * i / steps
                self._ink()
                self._dirty = True
            if animate:
                self._sleep(step_time)
        with self._lock:
            self.s = target
            self._ink()
            self._dirty = True

    # ================= running programs in the background =================

    def run_in_background(self, fn):
        """Start fn in the worker thread; returns immediately."""
        if self._worker is not None and self._worker.is_alive():
            print("Arm is busy - wait until it finishes.")
            return
        self._worker = threading.Thread(target=self._worker_main, args=(fn,),
                                        daemon=True)
        self._worker.start()

    def _worker_main(self, fn):
        self._clock = time.perf_counter()   # schedule starts now
        try:
            fn()
        except StopProgram:
            pass
        except Exception:
            traceback.print_exc()

    @property
    def busy(self):
        return self._worker is not None and self._worker.is_alive()

    def _on_close(self, _):
        self._stop = True
        self._timer.stop()

    # ================= window (GUI thread only) =================
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
            b.on_clicked(lambda e: self.run_in_background(fn))
            return b
        self.buttons = [
            button(0.72, 0.20, "Pen Up", self.pen_up),
            button(0.85, 0.20, "Pen Down", self.pen_down),
            button(0.72, 0.13, "Home", self.home),
            button(0.85, 0.13, "Clear paper", self.clear),
            button(0.72, 0.06, "Run program", lambda: my_program(self)),
        ]

    def _on_slider(self, _):
        if self._syncing or self.busy:
            return
        self._move_to(*(sl.val for sl in self.sliders), animate=False)

    def _on_timer(self):
        if self._dirty:
            self._render()

    def _render(self):
        with self._lock:
            s1, s2, s3 = self.s
            ink_x = np.array(self.ink_x, dtype=float)
            ink_y = np.array(self.ink_y, dtype=float)
            self._dirty = False

        elbow, pen = self._pen_xy(s1, s2)
        down = s3 <= PEN_CONTACT_ANGLE
        self.arm_line.set_data([0, elbow[0], pen[0]], [0, elbow[1], pen[1]])
        self.joints.set_data([0, elbow[0]], [0, elbow[1]])
        self.pen_dot.center = pen
        self.pen_dot.set_facecolor("navy" if down else "white")
        self.ink_line.set_data(ink_x, ink_y)
        h = 0 if down else min(3.0, (s3 - PEN_CONTACT_ANGLE) / 15.0)
        self.pen_side.set_data([0, 0], [h, h + 1.0])
        self.info.set_text(f"Servo 1 : {s1:4.0f} deg\n"
                           f"Servo 2 : {s2:4.0f} deg\n"
                           f"Servo 3 : {s3:4.0f} deg\n\n"
                           f"Pen     : {'DOWN' if down else 'UP'}")

        # move slider handles to match, without triggering _on_slider
        self._syncing = True
        for sl, v in zip(self.sliders, (s1, s2, s3)):
            if sl.val != v:
                sl.eventson = False
                sl.drawon = False
                sl.set_val(v)
                sl.eventson = True
                sl.drawon = True
        self._syncing = False

        self.fig.canvas.draw_idle()


# =====================================================================
# YOUR PROGRAM - runs when you press "Run program"
# Same calls you will later send to the Arduino.
# (It runs in a background thread, so the window stays responsive.)
# =====================================================================


def my_program(arm):
    arm.clear()

    with open("drawing_paths.json", "r") as file:
        paths = json.load(file)

    print(f"Drawing portrait using {len(paths)} paths...")

    arm.draw_paths(paths)

    arm.pen_up()
    arm.home()

    print("Portrait completed!")


if __name__ == "__main__":
    arm = ArmSimulator()
    plt.show()