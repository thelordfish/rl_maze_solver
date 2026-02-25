#exploration_strategies.py
"""
Exploration strategies for action selection.

These answer: "given current Q-values, which action do I pick?"
They are independent of the learning algorithm (MC, SARSA, Q-learning).
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np


class EpsilonGreedy:
    """
    With probability epsilon: pick a random action (explore).
    Otherwise: pick the best action (exploit).
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon

    def choose(self, q_values: Dict[str, float]) -> str:
        if random.random() < self.epsilon:
            return random.choice(list(q_values.keys()))
        max_val = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_val]
        return random.choice(best_actions)

    def get_probs(self, q_values: Dict[str, float]) -> Dict[str, float]:
        actions = list(q_values.keys())
        n = len(actions)
        max_val = max(q_values.values())

        # All tied → uniform
        if all(v == max_val for v in q_values.values()):
            return {a: 1.0 / n for a in actions}

        best_a = max(q_values, key=q_values.get)
        probs = {}
        for a in actions:
            if a == best_a:
                probs[a] = 1.0 - self.epsilon + (self.epsilon / n)
            else:
                probs[a] = self.epsilon / n
        return probs


class Softmax:
    """
    Convert Q-values into probabilities via Boltzmann distribution.
    Lower tau = more greedy, higher tau = more random.
    """

    def __init__(self, tau: float = 1.0) -> None:
        self.tau = tau

    def choose(self, q_values: Dict[str, float]) -> str:
        probs = self.get_probs(q_values)
        actions = list(probs.keys())
        weights = [probs[a] for a in actions]
        return random.choices(actions, weights=weights)[0]

    def get_probs(self, q_values: Dict[str, float]) -> Dict[str, float]:
        """Return the full probability distribution (for the visualiser)."""
        max_q = max(q_values.values())
        exp_q = {a: np.exp((q - max_q) / self.tau) for a, q in q_values.items()}
        z = sum(exp_q.values())
        return {a: exp_q[a] / z for a in q_values}