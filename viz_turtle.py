# viz_turtle.py
"""
Turtle-based maze visualiser with tkinter control panel.

Responsibilities:
  - Draw maze walls, start/goal markers, policy arrows
  - Animate the agent
  - Provide UI controls (strategy, algorithm, sliders, start/reset)

No RL logic lives here.
"""

import turtle
import tkinter as tk
from tkinter import font as tkfont
from typing import Dict, Tuple

from mdp_env import ACTIONS, DX, DY, State

HEADING = {"N": 90, "E": 0, "S": 270, "W": 180}


class TurtleMazeViz:

    def __init__(self, width: int, height: int, walls: Dict,
                 cell_size_pixels: int = 100):
        self.width = width
        self.height = height
        self.walls = walls
        self.cell_size = cell_size_pixels
        self._last_Q = None                     # for redraw after resize

        # --- Window setup ---
        win_w = self.width * self.cell_size + 100
        win_h = self.height * self.cell_size + 160

        self.screen = turtle.Screen()
        self.screen.title("RL Maze View")
        self.screen.setup(width=win_w, height=win_h)
        self.screen.screensize(win_w - 50, win_h - 50)
        self.screen.tracer(0, 0)

        # --- Turtle pens ---
        self.wall_pen    = turtle.Turtle(visible=False)
        self.overlay_pen = turtle.Turtle(visible=False)

        self.agent = turtle.Turtle(shape="triangle")
        self.agent.shapesize(0.6, 0.6)
        self.agent.color("red")
        self.agent.penup()

        # --- Initial draw ---
        self._recalc_layout()
        self._draw_maze_walls()
        self._draw_markers()
        self.agent.goto(self.cell_center_pixel(0, 0))
        self.screen.update()

        # --- Resize binding ---
        self._last_size = (win_w, win_h)
        self.screen.getcanvas().bind("<Configure>", self._on_resize)

    # ================================================================
    # Layout
    # ================================================================

    def _recalc_layout(self):
        """Derive cell_size and origin from current canvas dimensions."""
        canvas = self.screen.getcanvas()
        margin = 60
        usable_w = max(canvas.winfo_width()  - margin, 100)
        usable_h = max(canvas.winfo_height() - margin, 100)

        self.cell_size = min(usable_w // self.width, usable_h // self.height)
        self.origin_x  = -(self.width  * self.cell_size) / 2
        self.origin_y  = -(self.height * self.cell_size) / 2

    def _on_resize(self, event):
        new_size = (event.width, event.height)
        if new_size == self._last_size:
            return
        self._last_size = new_size
        self._recalc_layout()
        self._redraw_all()

    def _redraw_all(self):
        self.wall_pen.clear()
        self.overlay_pen.clear()
        self._draw_maze_walls()
        self._draw_markers()
        if self._last_Q is not None:
            self.draw_policy(self._last_Q)
        self.screen.update()

    # ================================================================
    # Drawing helpers
    # ================================================================

    def cell_center_pixel(self, x: int, y: int) -> Tuple[float, float]:
        px = self.origin_x + (x + 0.5) * self.cell_size
        py = self.origin_y + (y + 0.5) * self.cell_size
        return px, py

    def _draw_maze_walls(self):
        pen = self.wall_pen
        pen.pensize(2)
        pen.color("black")
        cs = self.cell_size

        for (x, y), dirs in self.walls.items():
            xl = self.origin_x + x * cs
            xr = xl + cs
            yb = self.origin_y + y * cs
            yt = yb + cs

            segments = [
                ("N", (xl, yt), (xr, yt)),
                ("E", (xr, yb), (xr, yt)),
                ("S", (xl, yb), (xr, yb)),
                ("W", (xl, yb), (xl, yt)),
            ]
            for d, start, end in segments:
                if dirs.get(d):
                    pen.penup();   pen.goto(start)
                    pen.pendown(); pen.goto(end)

        self.screen.update()

    def _draw_markers(self):
        markers = [((0, 0), "blue"),
                   ((self.width - 1, self.height - 1), "green")]
        for coord, colour in markers:
            cx, cy = self.cell_center_pixel(*coord)
            self.wall_pen.penup()
            self.wall_pen.color(colour)
            self.wall_pen.goto(cx, cy - 5)
            self.wall_pen.dot(15)
        self.screen.update()

    # ================================================================
    # Agent + policy overlay
    # ================================================================

    def move_agent(self, s: State, action: str = None, delay: float = 0.001):
        if action and action in HEADING:
            self.agent.setheading(HEADING[action])
        self.agent.goto(self.cell_center_pixel(s.x, s.y))
        self.screen.update()
        if delay > 0:
            self.screen._root.update()
            self.screen._root.after(int(delay * 1000))

    def draw_policy(self, Q: Dict):
        """Draw an arrow in each cell pointing in the greedy direction."""
        self._last_Q = Q
        pen = self.overlay_pen
        pen.clear()
        pen.color("purple")
        arrow_len = self.cell_size * 0.25

        for (x, y), q_dict in Q.items():
            best_a = max(q_dict, key=q_dict.get)
            cx, cy = self.cell_center_pixel(x, y)
            pen.penup();   pen.goto(cx, cy)
            pen.pendown();  pen.goto(cx + DX[best_a] * arrow_len,
                                     cy + DY[best_a] * arrow_len)
        self.screen.update()

    # ================================================================
    # Tkinter control panel
    # ================================================================

    def setup_controls(self, on_strategy_change, on_algorithm_change,
                       on_reset, on_start):
        root = self.screen._root
        font = tkfont.Font(family="Helvetica", size=11)

        frame = tk.Frame(root)
        frame.pack(side="bottom", fill="x", padx=10, pady=8)

        # --- Strategy radio buttons ---
        tk.Label(frame, text="Strategy:", font=font).pack(side="left", padx=(0, 4))
        self.strategy_var = tk.StringVar(value="softmax")

        def _on_strategy_switch():
            name = self.strategy_var.get()
            self._update_slider_for_strategy(name)
            on_strategy_change(name)

        for name in ["EpsilonGreedy", "Softmax"]:
            tk.Radiobutton(frame, text=name, variable=self.strategy_var,
                        value=name, font=font,
                        command=_on_strategy_switch).pack(side="left")

        # --- Parameter slider (shared for ε / τ) ---
        self.param_label = tk.Label(frame, text="  τ:", font=font)
        self.param_label.pack(side="left", padx=(8, 0))

        self.param_slider = tk.Scale(
            frame, from_=0.01, to=5.0, resolution=0.01,
            orient="horizontal", length=120, font=font)
        self.param_slider.set(2.5)
        self.param_slider.pack(side="left")

        # --- Algorithm radio buttons ---
        tk.Label(frame, text="    Algorithm:", font=font).pack(side="left", padx=(8, 4))
        self.algorithm_var = tk.StringVar(value="monte_carlo")

        for name in ["MonteCarlo", "SARSA", "QLearning"]:
            tk.Radiobutton(frame, text=name, variable=self.algorithm_var,
                        value=name, font=font,
                        command=lambda: on_algorithm_change(self.algorithm_var.get())
            ).pack(side="left")

        # --- Start / Reset buttons ---
        tk.Button(frame, text="Reset", font=font, command=on_reset,
                  bg="#ef4444", fg="white", padx=8).pack(side="right", padx=(8, 0))

        self.start_button = tk.Button(
            frame, text="Start", font=font, command=on_start,
            bg="#22c55e", fg="white", padx=8)
        self.start_button.pack(side="right", padx=(8, 0))

        # --- Status label ---
        self.status_label = tk.Label(frame, text="", fg="gray", font=font)
        self.status_label.pack(side="right")

    def _update_slider_for_strategy(self, name: str):
        if name == "epsilon_greedy":
            self.param_label.config(text="  ε:")
            self.param_slider.config(from_=0.01, to=1.0, resolution=0.01)
            self.param_slider.set(0.2)
        elif name == "softmax":
            self.param_label.config(text="  τ:")
            self.param_slider.config(from_=0.01, to=5.0, resolution=0.01)
            self.param_slider.set(1.0)

    def update_status(self, text: str):
        self.status_label.config(text=text)