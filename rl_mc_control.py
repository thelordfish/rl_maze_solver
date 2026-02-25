"""
rl_mc_control.py

Orchestrates:
- Environment setup (maze + MDP)
- Visualisation (turtle + policy charts)
- Learning loop (Monte Carlo control)
- UI controls (strategy, algorithm, reset)
"""

import numpy as np
import random
import matplotlib.pyplot as plt

from maze_gen import generate_prim_maze
from mdp_env import ACTIONS, GridMazeMDP
from viz_turtle import TurtleMazeViz
from policy_visualiser import PolicyVisualiser
from exploration_strategies import EpsilonGreedy, Softmax


class MazeRunner:
    """
    Owns the environment, visualisers, Q-table, and learning loop.
    No globals needed — everything lives here.
    """

    def __init__(self, width=6, height=6, seed=42, gamma=0.99,
                 step_cap=700, num_episodes=100):
        # Config
        self.width = width
        self.height = height
        self.gamma = gamma
        self.step_cap = step_cap
        self.num_episodes = num_episodes

        # Environment
        self.maze = generate_prim_maze(width, height, seed=seed)
        self.mdp = GridMazeMDP(width, height, self.maze.walls)

        # Strategy (swappable via controls)
        self.strategy = Softmax(tau=2.5)

        # Learning state
        self.Q = None
        self.returns = None
        self.reset_learning()

        # Visualisers
        self.turtle_viz = TurtleMazeViz(width, height, self.maze.walls)
        self.policy_viz = PolicyVisualiser(width, height)

        # Control flags
        self._reset_requested = False
        self._started = False

        # UI controls
        self.turtle_viz.setup_controls(
            on_strategy_change=self._on_strategy_change,
            on_algorithm_change=self._on_algorithm_change,
            on_reset=self._on_reset,
            on_start=self._on_start
        )

    def _on_start(self):
        self._started = True
        self.turtle_viz.start_button.config(state="disabled", bg="gray")

    # ---- Learning state ----

    def reset_learning(self):
        """Zero out Q-table and return history."""
        self.Q = {
            (x, y): {a: 0.0 for a in ACTIONS}
            for x in range(self.width)
            for y in range(self.height)
        }
        self.returns = {
            (x, y, a): []
            for x in range(self.width)
            for y in range(self.height)
            for a in ACTIONS
        }

    # ---- UI callbacks ----

    def _on_strategy_change(self, name):
        slider_val = self.turtle_viz.param_slider.get()
        if name == "epsilon_greedy":
            self.strategy = EpsilonGreedy(epsilon=slider_val)
        elif name == "softmax":
            self.strategy = Softmax(tau=slider_val)

    def _on_algorithm_change(self, name):
        print(f"Algorithm: {name}")  # placeholder until SARSA/Q-learning added

    def _on_reset(self):
        self._reset_requested = True

    def _sync_slider_to_strategy(self):
        """Read the current slider value into the active strategy."""
        val = self.turtle_viz.param_slider.get()
        if isinstance(self.strategy, EpsilonGreedy):
            self.strategy.epsilon = val
        elif isinstance(self.strategy, Softmax):
            self.strategy.tau = val

    # ---- Policy helper ----

    def _get_policy_map(self):
        """Convert Q-table into probability dict for the visualiser."""
        return {
            (x, y): self.strategy.get_probs(self.Q[(x, y)])
            for x in range(self.width)
            for y in range(self.height)
        }

    # ---- Episode logic ----

    def _generate_episode(self):
        """
        Run one episode: agent walks from start until goal or step cap.
        Returns list of (state, action, reward) tuples.
        """
        episode = []
        s = self.mdp.start()

        while not self.mdp.is_terminal(s):
            a = self.strategy.choose(self.Q[(s.x, s.y)])
            s_next = self.mdp.transition(s, a)
            r = self.mdp.reward(s, a, s_next)

            episode.append(((s.x, s.y), a, r))
            s = s_next

            self.turtle_viz.move_agent(s, action=a, delay=0.001)

            if len(episode) > self.step_cap:
                break

        return episode

    def _learn_from_episode(self, episode):
        """
        First-visit Monte Carlo update:
        Walk backwards through the episode, accumulating returns.
        """
        G = 0
        visited = set()

        for i in range(len(episode) - 1, -1, -1):
            (x, y), a, r = episode[i]
            G = r + self.gamma * G

            if ((x, y), a) not in visited:
                visited.add(((x, y), a))
                self.returns[(x, y, a)].append(G)
                self.Q[(x, y)][a] = np.mean(self.returns[(x, y, a)])

    def _update_visualisers(self, episode_num, episode_length):
        """Push current policy to both visualisers."""
        policy = self._get_policy_map()
        self.policy_viz.update(policy)
        self.turtle_viz.draw_policy(self.Q)
        self.turtle_viz.update_status(
            f"Episode {episode_num}  |  Steps: {episode_length}"
        )
        plt.pause(0.01)

    # ---- Main loop ----

    def run(self):
        """Run the full training loop."""
        plt.ion()

    # Wait for Start button
        while not self._started:
            self.turtle_viz.screen._root.update()
            plt.pause(0.05)

        ep = 0
        while ep < self.num_episodes:

            # Handle reset
            if self._reset_requested:
                self.reset_learning()
                ep=0
                self._reset_requested = False
                self._update_visualisers(ep, 0)  # <-- add this
                self.turtle_viz.start_button.config(state="normal", bg="#22c55e")
                self._started = False
                print("Reset!")
                            # Wait for Start again
                while not self._started:
                    self.turtle_viz.screen._root.update()
                    plt.pause(0.05)

            # Pick up live slider changes
            self._sync_slider_to_strategy()

            # Run one episode
            episode = self._generate_episode()
            self._learn_from_episode(episode)
            self._update_visualisers(ep, len(episode))

            print(f"Episode {ep} completed. ({len(episode)} steps)")
            ep+=1

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    runner = MazeRunner(
        width=6,
        height=6,
        seed=42,
        gamma=0.99,
        step_cap=500,
        num_episodes=100
    )
    runner.run()