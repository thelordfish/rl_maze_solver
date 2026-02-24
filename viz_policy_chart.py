# viz_policy_chart.py
"""
viz_policy_chart.py

Purpose:
--------
Live-updating matplotlib bar chart to visualise a policy π(a|s)
for the *current* state s.

This module does NOT:
- implement RL
- implement the MDP
- generate mazes
- use turtle

It ONLY:
- shows a bar chart for actions {N,E,S,W}
- updates bar heights when you call update(...)
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt

from mdp_env import ACTIONS, State


class PolicyBarChart:
    """
    A tiny live chart:

      x-axis: actions N/E/S/W
      y-axis: probability π(a|s)

    update(s, probs) updates the figure in-place.
    """

    def __init__(self) -> None:
        plt.ion()  # interactive mode ON

        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_ylabel("π(a|s)")
        self.title_text = self.ax.set_title("Policy at state s = (?, ?)")

        # Start uniform
        initial_probs = [0.25, 0.25, 0.25, 0.25]
        self.bars = self.ax.bar(list(ACTIONS), initial_probs)

        # Numeric labels above bars
        self.labels = []
        for bar in self.bars:
            txt = self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "",
                ha="center",
                va="bottom",
            )
            self.labels.append(txt)

        self._redraw()

    def update(self, s: State, probs: Dict[str, float]) -> None:
        """
        Update the plot to show π(a|s) for the provided state.
        probs must contain keys: N,E,S,W.
        """
        self.title_text.set_text(f"Policy at state s = ({s.x}, {s.y})")

        for i, a in enumerate(ACTIONS):
            p = float(probs[a])
            self.bars[i].set_height(p)

            # Move and update numeric label
            self.labels[i].set_y(p)
            self.labels[i].set_text(f"{p:.2f}")

        self._redraw()

    def _redraw(self) -> None:
        """
        Force matplotlib to repaint immediately.
        (We use both flush_events and a tiny pause for backend compatibility.)
        """
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# -----------------------------
# Quick standalone test
# -----------------------------
if __name__ == "__main__":
    import time
    import random

    chart = PolicyBarChart()
    rng = random.Random(0)

    s = State(0, 0)

    for t in range(120):
        # Create some fake probabilities that move around over time
        raw = [rng.random() + 0.2 for _ in ACTIONS]
        z = sum(raw)
        probs = {a: raw[i] / z for i, a in enumerate(ACTIONS)}

        s = State(t % 8, (t // 8) % 8)
        chart.update(s, probs)

        time.sleep(0.05)

    # Keep window open after loop
    plt.ioff()
    plt.show()