import turtle
import time
from typing import Dict, Tuple

from mdp_env import ACTIONS, DX, DY, State


class TurtleMazeViz:
    def __init__(self, width: int, height: int, walls: Dict, cell_size_pixels: int = 100):
        self.width = width
        self.height = height
        self.walls = walls
        self.cell_size = cell_size_pixels

        # Store latest Q for redrawing policy overlay after resize
        self._last_Q = None

        # Size the window to fit the maze
        win_w = self.width * self.cell_size + 100
        win_h = self.height * self.cell_size + 160

        self.screen = turtle.Screen()
        self.screen.title("RL Maze View")
        self.screen.setup(width=win_w, height=win_h)
        self.screen.screensize(win_w - 50, win_h - 50)
        self.screen.tracer(0, 0)

        self.wall_pen = turtle.Turtle(visible=False)
        self.overlay_pen = turtle.Turtle(visible=False)

        self.agent = turtle.Turtle(shape="triangle")
        self.agent.shapesize(0.6, 0.6)
        self.agent.color("red")
        self.agent.penup()

        self._recalc_layout()
        self._draw_maze_walls()
        self._draw_markers()

        #Position agent at start
        self.agent.goto(self.cell_center_pixel(0,0))
        self.screen.update()

        # Bind resize
        self._last_size = (win_w, win_h)
        self.screen.getcanvas().bind("<Configure>", self._on_resize)

    # ---- Layout / sizing ----

    def _recalc_layout(self):
        """Recalculate cell size and origin from current canvas size."""
        canvas = self.screen.getcanvas()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        margin = 60
        usable_w = max(cw - margin, 100)
        usable_h = max(ch - margin, 100)

        self.cell_size = min(usable_w // self.width, usable_h // self.height)
        self.origin_x = -((self.width * self.cell_size) / 2)
        self.origin_y = -((self.height * self.cell_size) / 2)

    def _on_resize(self, event):
        """Called when the window is resized."""
        new_size = (event.width, event.height)
        if new_size == self._last_size:
            return
        self._last_size = new_size

        self._recalc_layout()
        self._redraw_all()

    def _redraw_all(self):
        """Clear everything and redraw at current size."""
        self.wall_pen.clear()
        self.overlay_pen.clear()

        self._draw_maze_walls()
        self._draw_markers()

        if self._last_Q is not None:
            self.draw_policy(self._last_Q)

        self.screen.update()

    # ---- Drawing ----

    def cell_center_pixel(self, x: int, y: int) -> Tuple[float, float]:
        px = self.origin_x + x * self.cell_size + self.cell_size / 2
        py = self.origin_y + y * self.cell_size + self.cell_size / 2
        return px, py

    def _draw_maze_walls(self):
        self.wall_pen.pensize(2)
        self.wall_pen.color("black")
        cs = self.cell_size
        for (x, y), directions in self.walls.items():
            xl, xr = self.origin_x + x * cs, self.origin_x + (x + 1) * cs
            yb, yt = self.origin_y + y * cs, self.origin_y + (y + 1) * cs

            segs = [("N", (xl, yt), (xr, yt)), ("E", (xr, yb), (xr, yt)),
                    ("S", (xl, yb), (xr, yb)), ("W", (xl, yb), (xl, yt))]

            for d, start, end in segs:
                if directions.get(d):
                    self.wall_pen.penup()
                    self.wall_pen.goto(start)
                    self.wall_pen.pendown()
                    self.wall_pen.goto(end)
        self.screen.update()

    def _draw_markers(self):
        for (coord, color) in [((0, 0), "blue"), ((self.width - 1, self.height - 1), "green")]:
            cx, cy = self.cell_center_pixel(*coord)
            self.wall_pen.penup()
            self.wall_pen.color(color)
            self.wall_pen.goto(cx, cy - 5)
            self.wall_pen.dot(15)
        self.screen.update()


    def move_agent(self, s, action=None, delay=0.001):
        HEADING = {"N": 90, "E": 0, "S": 270, "W": 180}

        if action and action in HEADING:
            self.agent.setheading(HEADING[action])
        self.agent.goto(self.cell_center_pixel(s.x, s.y))
        self.screen.update()
        if delay > 0:
            self.screen._root.update()
            self.screen._root.after(int(delay * 1000))

    def draw_policy(self, Q):
        """Draws arrows showing the current best action in each cell."""
        self._last_Q = Q
        self.overlay_pen.clear()
        self.overlay_pen.color("purple")
        arrow_len = self.cell_size * 0.25  # scale arrows to cell size
        for (x, y), q_dict in Q.items():
            best_a = max(q_dict, key=q_dict.get)
            cx, cy = self.cell_center_pixel(x, y)
            self.overlay_pen.penup()
            self.overlay_pen.goto(cx, cy)
            self.overlay_pen.pendown()
            self.overlay_pen.goto(cx + DX[best_a] * arrow_len,
                                  cy + DY[best_a] * arrow_len)
        self.screen.update()

    # ---- Controls ----

    def setup_controls(self, on_strategy_change, on_algorithm_change, on_reset, on_start):
        """Add a tkinter control panel below the maze."""
        import tkinter as tk
        from tkinter import font as tkfont

        root = self.screen._root
        bigger = tkfont.Font(family="Helvetica", size=11)

        frame = tk.Frame(root)
        frame.pack(side="bottom", fill="x", padx=10, pady=8)

        # --- Strategy selector ---
        tk.Label(frame, text="Strategy:", font=bigger).pack(side="left", padx=(0, 4))
        self.strategy_var = tk.StringVar(value="softmax")

        def _on_strategy_switch():
            name = self.strategy_var.get()
            self._update_slider_for_strategy(name)
            on_strategy_change(name)

        for text, val in [("ε-Greedy", "epsilon_greedy"), ("Softmax", "softmax")]:
            tk.Radiobutton(frame, text=text, variable=self.strategy_var,
                           value=val, font=bigger,
                           command=_on_strategy_switch).pack(side="left")

        # --- Single unified slider ---
        self.param_label = tk.Label(frame, text="  ε:", font=bigger)
        self.param_label.pack(side="left", padx=(8, 0))
        self.param_slider = tk.Scale(frame, from_=0.01, to=1.0, resolution=0.01,
                                     orient="horizontal", length=120, font=bigger)
        self.param_slider.set(2.5)
        self.param_slider.pack(side="left")
        self._update_slider_for_strategy("softmax")  
        self.param_slider.set(2.5)  


        # Convenience accessors so rl_mc_control.py works unchanged
        self.epsilon_slider = self.param_slider
        self.tau_slider = self.param_slider

        # --- Algorithm selector ---
        tk.Label(frame, text="    Algorithm:", font=bigger).pack(side="left", padx=(8, 4))
        self.algorithm_var = tk.StringVar(value="monte_carlo")
        for text, val in [("MC", "monte_carlo"), ("SARSA", "sarsa"), ("Q-Learn", "q_learning")]:
            tk.Radiobutton(frame, text=text, variable=self.algorithm_var,
                           value=val, font=bigger,
                           command=lambda: on_algorithm_change(self.algorithm_var.get())
                           ).pack(side="left")

        # --- Reset button ---
        tk.Button(frame, text="Reset", font=bigger, command=on_reset,
                bg="#ef4444", fg="white", padx=8).pack(side="right", padx=(8, 0))
        
        # --- Start button ---
        self.start_button = tk.Button(frame, text="Start", font=bigger,
                                       command=on_start, bg="#22c55e", fg="white", padx=8)
        self.start_button.pack(side="right", padx=(8, 0))

        # --- Status label ---
        self.status_label = tk.Label(frame, text="", fg="gray", font=bigger)
        self.status_label.pack(side="right")

    def _update_slider_for_strategy(self, name):
        """Switch the slider label and range to match the active strategy."""
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