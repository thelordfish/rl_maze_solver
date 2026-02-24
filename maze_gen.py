# maze_gen.py
"""
maze_gen.py

Purpose:
--------
Generate an 8x8 (or arbitrary size) maze using mazelib's Randomized Prim
algorithm and convert it into an explicit per-cell wall representation
suitable for a Markov Decision Process.

We deliberately convert mazelib's internal grid representation into a
clear structure:

    walls[(x, y)][direction] = True   -> wall exists (blocked)
    walls[(x, y)][direction] = False  -> passage open

This file contains NO reinforcement learning logic.
It only produces structured environment data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

from mazelib import Maze
from mazelib.generate.Prims import Prims


# ------------------------------------------------------------
# Basic movement definitions (shared with the MDP later)
# ------------------------------------------------------------

# Allowed action directions
ACTIONS = ("N", "E", "S", "W")

# Movement delta for each direction
# If agent is at (x, y) and moves in direction d:
#   new_x = x + DX[d]
#   new_y = y + DY[d]
DX = {
    "N": 0,
    "E": 1,
    "S": 0,
    "W": -1,
}

DY = {
    "N": 1,
    "E": 0,
    "S": -1,
    "W": 0,
}

# Opposite directions (for symmetry enforcement)
OPPOSITE_DIRECTION = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
}


# ------------------------------------------------------------
# Explicit data container for maze output
# ------------------------------------------------------------

@dataclass(frozen=True)
class MazeData:
    width: int
    height: int

    # walls[(x,y)] -> dict of direction->bool
    # True  = wall present
    # False = open passage
    walls: Dict[Tuple[int, int], Dict[str, bool]]


# ------------------------------------------------------------
# Maze generation
# ------------------------------------------------------------

def generate_prim_maze(
    width: int,
    height: int,
    seed: int | None = None
) -> MazeData:
    """
    Generate a perfect maze using Randomized Prim (via mazelib).

    Steps:
    ------
    1. Ask mazelib to generate a maze grid.
    2. mazelib returns a "pixel-style" grid:
           - 1 means wall
           - 0 means open corridor
    3. We convert that representation into per-cell wall flags.
    """

    if width < 2 or height < 2:
        raise ValueError("width and height must both be >= 2")

    if seed is not None:
        random.seed(seed)

    # --------------------------------------------------------
    # Step 1: Use mazelib to generate maze
    # --------------------------------------------------------

    maze_object = Maze()
    maze_object.generator = Prims(width, height)
    maze_object.generate()

    # mazelib grid dimensions:
    #   rows = 2*height + 1
    #   cols = 2*width + 1
    # Cells are located at:
    #   (row = 2*y + 1, col = 2*x + 1)
    #
    # Between two adjacent cells there is exactly one grid location
    # that represents the "wall slot" between them.

    raw_grid = maze_object.grid

    # --------------------------------------------------------
    # Step 2: Initialise wall dictionary (all walls present)
    # --------------------------------------------------------

    walls: Dict[Tuple[int, int], Dict[str, bool]] = {}

    for y in range(height):
        for x in range(width):
            walls[(x, y)] = {
                "N": True,
                "E": True,
                "S": True,
                "W": True,
            }

    # --------------------------------------------------------
    # Step 3: Inspect raw_grid to determine open passages
    # --------------------------------------------------------

    for cell_y in range(height):
        for cell_x in range(width):

            # Locate the center of this cell in raw_grid
            grid_row_of_cell_center = 2 * cell_y + 1
            grid_col_of_cell_center = 2 * cell_x + 1

            # ------------------------
            # Check NORTH connectivity
            # ------------------------
            north_row = grid_row_of_cell_center + 1
            north_col = grid_col_of_cell_center

            if raw_grid[north_row][north_col] == 0:
                walls[(cell_x, cell_y)]["N"] = False

            # ------------------------
            # Check SOUTH connectivity
            # ------------------------
            south_row = grid_row_of_cell_center - 1
            south_col = grid_col_of_cell_center

            if raw_grid[south_row][south_col] == 0:
                walls[(cell_x, cell_y)]["S"] = False

            # ------------------------
            # Check EAST connectivity
            # ------------------------
            east_row = grid_row_of_cell_center
            east_col = grid_col_of_cell_center + 1

            if raw_grid[east_row][east_col] == 0:
                walls[(cell_x, cell_y)]["E"] = False

            # ------------------------
            # Check WEST connectivity
            # ------------------------
            west_row = grid_row_of_cell_center
            west_col = grid_col_of_cell_center - 1

            if raw_grid[west_row][west_col] == 0:
                walls[(cell_x, cell_y)]["W"] = False

    # --------------------------------------------------------
    # Step 4: Enforce symmetry (safety step)
    # --------------------------------------------------------
    #
    # If (x,y) says EAST is open,
    # then (x+1,y) must say WEST is open.
    #
    # We enforce this explicitly to avoid subtle bugs.

    for y in range(height):
        for x in range(width):
            for direction in ACTIONS:
                if walls[(x, y)][direction] is False:

                    neighbour_x = x + DX[direction]
                    neighbour_y = y + DY[direction]

                    if 0 <= neighbour_x < width and 0 <= neighbour_y < height:
                        opposite = OPPOSITE_DIRECTION[direction]
                        walls[(neighbour_x, neighbour_y)][opposite] = False

    # --------------------------------------------------------
    # Return structured MazeData
    # --------------------------------------------------------

    return MazeData(
        width=width,
        height=height,
        walls=walls
    )


# ------------------------------------------------------------
# Quick standalone test
# ------------------------------------------------------------

if __name__ == "__main__":
    maze_data = generate_prim_maze(8, 8, seed=123)

    print("Maze generated.")
    print("Dimensions:", maze_data.width, "x", maze_data.height)
    print("Walls at (0,0):", maze_data.walls[(0, 0)])