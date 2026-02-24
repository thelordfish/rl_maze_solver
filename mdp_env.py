# mdp_env.py
"""
mdp_env.py

Purpose:
--------
Define an explicit Markov Decision Process (MDP) on top of a grid maze.

This file is intentionally "textbook":
- State space: (x, y) grid coordinates
- Action space: {N, E, S, W}
- Transition function: T(s, a) -> s' (deterministic here)
- Reward function: R(s, a, s') -> float
- Terminal condition: reaching the goal cell

No RL algorithm is implemented here.
No visualisation is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

# We keep the same action naming convention as maze_gen.py
ACTIONS = ("N", "E", "S", "W")

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


@dataclass(frozen=True)
class State:
    """
    A state is simply the agent's cell position in the grid.
    """
    x: int
    y: int


class GridMazeMDP:
    """
    Explicit deterministic MDP for navigating a maze.

    Parameters:
    ----------
    width, height:
        Grid size in cells.

    walls:
        walls[(x, y)][direction] = True/False
        True  means a wall is present (movement blocked).
        False means passage is open (movement allowed).

    step_cost:
        The reward received each step. You specified -1.

    gamma:
        Discount factor used later by RL algorithms when computing returns.
        It's part of the "MDP + objective" setup.
    """

    def __init__(
        self,
        width: int,
        height: int,
        walls: Dict[Tuple[int, int], Dict[str, bool]],
        step_cost: float = -1.0,
        gamma: float = 0.99,
        start_state: State | None = None,
        goal_state: State | None = None,
        goal_reward: float = 100.0,
        wall_bump_cost: float = -2.0
    ) -> None:
        self.width = width
        self.height = height
        self.walls = walls

        self.step_cost = float(step_cost)
        self.gamma = float(gamma)
        self.goal_reward = float(goal_reward)   
        self.wall_bump_cost = float(wall_bump_cost)
        # Default start/goal if not provided
        self._start_state = start_state if start_state is not None else State(0, 0)
        self._goal_state = goal_state if goal_state is not None else State(width - 1, height - 1)

        # Basic validation (kept simple and explicit)
        self._validate_state_in_bounds(self._start_state, name="start_state")
        self._validate_state_in_bounds(self._goal_state, name="goal_state")

    # -----------------------------
    # State space helpers
    # -----------------------------

    def all_states(self) -> Iterable[State]:
        """
        Iterate over every possible state in the grid.
        Useful for full-policy visualisation later.
        """
        for y in range(self.height):
            for x in range(self.width):
                yield State(x, y)

    def start(self) -> State:
        """
        Return the start state s_0.
        """
        return self._start_state

    def goal(self) -> State:
        """
        Return the goal / terminal state.
        """
        return self._goal_state

    def is_terminal(self, s: State) -> bool:
        """
        Terminal condition:
        Episode ends when the agent reaches the goal cell.
        """
        return (s.x == self._goal_state.x) and (s.y == self._goal_state.y)

    def _validate_state_in_bounds(self, s: State, name: str) -> None:
        if not (0 <= s.x < self.width and 0 <= s.y < self.height):
            raise ValueError(f"{name} {s} is out of bounds for grid {self.width}x{self.height}")

    # -----------------------------
    # Action space
    # -----------------------------

    def all_actions(self) -> Tuple[str, str, str, str]:
        """
        Return the action set A (same for all states here).
        """
        return ACTIONS

    # -----------------------------
    # Transition function T(s,a)->s'
    # -----------------------------

    def transition(self, s: State, a: str) -> State:
        """
        Deterministic transition function.

        If the wall in direction a exists at state s, then movement is blocked:
            s' = s  (agent stays in place)

        Otherwise:
            s' = (s.x + DX[a], s.y + DY[a])
        """
        if a not in ACTIONS:
            raise ValueError(f"Unknown action '{a}'. Must be one of {ACTIONS}.")

        # If there is a wall, the agent cannot move: "bounce" in place.
        wall_present = self.walls[(s.x, s.y)][a]
        if wall_present:
            return s

        # Passage is open: attempt to move to the neighbor cell.
        next_x = s.x + DX[a]
        next_y = s.y + DY[a]

        # In a well-formed maze this should remain in bounds,
        # but we keep an explicit guard anyway.
        if not (0 <= next_x < self.width and 0 <= next_y < self.height):
            return s

        return State(next_x, next_y)

    # -----------------------------
    # Reward function R(s,a,s')
    # -----------------------------

    # mdp_env.py  (inside GridMazeMDP.reward)
    def reward(self, s: State, a: str, s_next: State) -> float:
        base_reward = self.step_cost

        # extra penalty if action was blocked (bounce in place)
        if (s_next.x == s.x) and (s_next.y == s.y):
            base_reward += self.wall_bump_cost

        if self.is_terminal(s_next):
            return base_reward + self.goal_reward

        return base_reward


# -----------------------------
# Quick standalone test
# -----------------------------
if __name__ == "__main__":
    # Minimal smoke test with a trivial "no interior walls removed" 2x2.
    width, height = 2, 2
    walls = {(x, y): {d: True for d in ACTIONS} for y in range(height) for x in range(width)}
    mdp = GridMazeMDP(width, height, walls)

    s0 = mdp.start()
    print("start:", s0)
    print("goal:", mdp.goal())
    print("terminal(start)?", mdp.is_terminal(s0))

    a = "E"
    s1 = mdp.transition(s0, a)
    r = mdp.reward(s0, a, s1)
    print("transition:", s0, "--", a, "-->", s1, "reward:", r)