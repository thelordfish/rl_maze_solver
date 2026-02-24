import turtle
import time
from typing import Dict, Tuple

# We only import the data structures we need from our other files
from mdp_env import ACTIONS, DX, DY, State

class TurtleMazeViz:
    def __init__(self, width: int, height: int, walls: Dict, cell_size_pixels: int = 60):
        self.width = width
        self.height = height
        self.walls = walls
        self.cell_size = cell_size_pixels

        self.screen = turtle.Screen()
        self.screen.title("RL Maze View")
        self.screen.tracer(0, 0) # This makes it draw instantly
        
        self.wall_pen = turtle.Turtle(visible=False)
        self.overlay_pen = turtle.Turtle(visible=False)
        self.agent = turtle.Turtle(shape="circle")
        self.agent.shapesize(0.6, 0.6)
        self.agent.penup()

        self.origin_x = -((self.width * self.cell_size) / 2)
        self.origin_y = -((self.height * self.cell_size) / 2)

        self._draw_maze_walls()
        self._draw_markers()

    def cell_center_pixel(self, x: int, y: int) -> Tuple[float, float]:
        px = self.origin_x + x * self.cell_size + self.cell_size / 2
        py = self.origin_y + y * self.cell_size + self.cell_size / 2
        return px, py

    def _draw_maze_walls(self):
        self.wall_pen.clear()
        self.wall_pen.pensize(2)
        cs = self.cell_size
        for (x, y), directions in self.walls.items():
            xl, xr = self.origin_x + x * cs, self.origin_x + (x + 1) * cs
            yb, yt = self.origin_y + y * cs, self.origin_y + (y + 1) * cs
            
            # Map directions to start/end coordinates
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
        # Blue Start (0,0), Green Goal (Top Right)
        for (coord, color) in [((0,0), "blue"), ((self.width-1, self.height-1), "green")]:
            cx, cy = self.cell_center_pixel(*coord)
            self.wall_pen.penup()
            self.wall_pen.color(color)
            self.wall_pen.goto(cx, cy - 5)
            self.wall_pen.dot(15)
        self.screen.update()

    def move_agent(self, s, delay=0.001):
        self.agent.goto(self.cell_center_pixel(s.x, s.y))
        self.screen.update()
        if delay > 0: time.sleep(delay)

    def draw_policy(self, Q):
        """Draws arrows showing the current best action in each cell."""
        self.overlay_pen.clear()
        self.overlay_pen.color("purple")
        for (x, y), q_dict in Q.items():
            best_a = max(q_dict, key=q_dict.get)
            cx, cy = self.cell_center_pixel(x, y)
            self.overlay_pen.penup()
            self.overlay_pen.goto(cx, cy)
            self.overlay_pen.pendown()
            self.overlay_pen.goto(cx + DX[best_a]*15, cy + DY[best_a]*15)
        self.screen.update()