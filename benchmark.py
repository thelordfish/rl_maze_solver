"""
benchmark.py

Runs all algorithm × strategy combinations over many mazes and episodes,
and reports which configuration learns fastest and most reliably.

No visualisation. No turtle. Just numbers.

Metrics per combination:
  - solve_rate:        % of episodes (across all runs) that reached the goal
  - first_solve_ep:    median episode number of the FIRST solve in each run
  - stable_ep:         median episode number where 5 consecutive solves occurred
  - steps_when_solved: mean steps taken on episodes that did solve
"""

import sys
import os
import random
import numpy as np
from statistics import median, mean

# ---- allow imports from same directory ----
sys.path.insert(0, os.path.dirname(__file__))

from maze_gen import generate_prim_maze
from mdp_env import ACTIONS, GridMazeMDP
from algorithms import MonteCarlo, SARSA, QLearning
from exploration_strategies import EpsilonGreedy, Softmax


# ============================================================
# CONFIG  — edit these
# ============================================================

MAZE_WIDTH      = 6
MAZE_HEIGHT     = 6
GAMMA           = 0.99
ALPHA           = 0.1          # for SARSA and Q-Learning
STEP_CAP        = MAZE_WIDTH * MAZE_HEIGHT * 3       # max steps before episode is abandoned
NUM_EPISODES    = 200          # episodes per run
NUM_RUNS        = 50           # independent runs per combination (different maze seeds)
STABLE_WINDOW   = 5            # how many consecutive solves = "stable"

# Exploration parameters to test
EPSILON         = 0.15
TAU             = 1.0


# ============================================================
# Helpers
# ============================================================

def make_fresh_Q(width, height):
    return {
        (x, y): {a: 0.0 for a in ACTIONS}
        for x in range(width)
        for y in range(height)
    }


def run_episode_mc(mdp, Q, strategy, algorithm, step_cap):
    """Full episode for Monte Carlo. Returns (solved, steps)."""
    episode = []
    s = mdp.start()
    solved = False

    for _ in range(step_cap):
        a = strategy.choose(Q[(s.x, s.y)])
        s_next = mdp.transition(s, a)
        r = mdp.reward(s, a, s_next)
        episode.append(((s.x, s.y), a, r))
        s = s_next
        if mdp.is_terminal(s):
            solved = True
            break

    algorithm.learn(Q, episode)
    return solved, len(episode)


def run_episode_td(mdp, Q, strategy, algorithm, step_cap):
    """Step-by-step episode for SARSA / Q-Learning. Returns (solved, steps)."""
    s = mdp.start()
    a = strategy.choose(Q[(s.x, s.y)])
    solved = False
    steps = 0

    for _ in range(step_cap):
        s_next = mdp.transition(s, a)
        r = mdp.reward(s, a, s_next)
        a_next = strategy.choose(Q[(s_next.x, s_next.y)])
        algorithm.learn(Q, (s.x, s.y), a, r, (s_next.x, s_next.y), a_next)
        s, a = s_next, a_next
        steps += 1
        if mdp.is_terminal(s):
            solved = True
            break

    return solved, steps


def first_stable_episode(solve_log, window):
    """
    Given a list of booleans (solved per episode),
    return the episode index where `window` consecutive Trues first occur.
    Returns None if it never happens.
    """
    count = 0
    for i, solved in enumerate(solve_log):
        if solved:
            count += 1
            if count >= window:
                return i - window + 1
        else:
            count = 0
    return None


def run_single_trial(seed, algo_name, strategy_name):
    """
    One full run: fresh maze, fresh Q, NUM_EPISODES episodes.
    Returns (solve_log, steps_log).
    """
    maze = generate_prim_maze(MAZE_WIDTH, MAZE_HEIGHT, seed=seed)
    mdp  = GridMazeMDP(MAZE_WIDTH, MAZE_HEIGHT, maze.walls, gamma=GAMMA)
    Q    = make_fresh_Q(MAZE_WIDTH, MAZE_HEIGHT)

    # Build algorithm
    if algo_name == "MonteCarlo":
        algo = MonteCarlo(gamma=GAMMA)
        use_mc = True
    elif algo_name == "SARSA":
        algo = SARSA(gamma=GAMMA, alpha=ALPHA)
        use_mc = False
    else:  # QLearning
        algo = QLearning(gamma=GAMMA, alpha=ALPHA)
        use_mc = False

    # Build strategy
    if strategy_name == "EpsilonGreedy":
        strategy = EpsilonGreedy(epsilon=EPSILON)
    else:
        strategy = Softmax(tau=TAU)

    solve_log = []
    steps_log = []

    for _ in range(NUM_EPISODES):
        if use_mc:
            solved, steps = run_episode_mc(mdp, Q, strategy, algo, STEP_CAP)
        else:
            solved, steps = run_episode_td(mdp, Q, strategy, algo, STEP_CAP)
        solve_log.append(solved)
        steps_log.append(steps)

    return solve_log, steps_log


# ============================================================
# Main benchmark loop
# ============================================================

COMBINATIONS = [
    ("MonteCarlo", "EpsilonGreedy"),
    ("MonteCarlo", "Softmax"),
    ("SARSA",      "EpsilonGreedy"),
    ("SARSA",      "Softmax"),
    ("QLearning",  "EpsilonGreedy"),
    ("QLearning",  "Softmax"),
]

def run_benchmark():
    print(f"\nBenchmark settings:")
    print(f"  Maze: {MAZE_WIDTH}x{MAZE_HEIGHT}  |  Episodes: {NUM_EPISODES}  |  Runs: {NUM_RUNS}  |  Step cap: {STEP_CAP}")
    print(f"  Epsilon: {EPSILON}  |  Tau: {TAU}  |  Alpha: {ALPHA}  |  Gamma: {GAMMA}\n")

    seeds = list(range(NUM_RUNS))  # reproducible, one seed per run
    results = {}

    for algo_name, strat_name in COMBINATIONS:
        label = f"{algo_name} + {strat_name}"
        print(f"Running: {label} ...", flush=True)

        all_solve_rates   = []
        first_solve_eps   = []
        stable_eps        = []
        solved_step_counts= []

        for seed in seeds:
            solve_log, steps_log = run_single_trial(seed, algo_name, strat_name)

            # Solve rate for this run
            all_solve_rates.append(mean(solve_log) * 100)

            # First solve
            first_solve = next((i for i, s in enumerate(solve_log) if s), None)
            if first_solve is not None:
                first_solve_eps.append(first_solve)

            # Stable solve
            stable = first_stable_episode(solve_log, STABLE_WINDOW)
            if stable is not None:
                stable_eps.append(stable)

            # Steps on solved episodes only
            for solved, steps in zip(solve_log, steps_log):
                if solved:
                    solved_step_counts.append(steps)

        results[label] = {
            "solve_rate":    mean(all_solve_rates),
            "first_solve_ep": median(first_solve_eps) if first_solve_eps else float("inf"),
            "stable_ep":      median(stable_eps)      if stable_eps      else float("inf"),
            "steps_solved":   mean(solved_step_counts) if solved_step_counts else float("inf"),
            "pct_runs_stable": 100 * len(stable_eps) / NUM_RUNS,
        }

    # ---- Print results table ----
    col_w = 26
    num_w = 12

    header = (
        f"{'Combination':<{col_w}}"
        f"{'Solve Rate %':>{num_w}}"
        f"{'First Solve Ep':>{num_w}}"
        f"{'Stable Ep':>{num_w}}"
        f"{'Steps (solved)':>{num_w}}"
        f"{'% Runs Stable':>{num_w}}"
    )
    divider = "-" * len(header)

    print(f"\n{divider}")
    print(header)
    print(divider)

    # Sort by stable episode (lower = learns fastest reliably)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["stable_ep"])

    for label, r in sorted_results:
        stable_str = f"{r['stable_ep']:.0f}" if r['stable_ep'] != float("inf") else "never"
        first_str  = f"{r['first_solve_ep']:.0f}" if r['first_solve_ep'] != float("inf") else "never"
        print(
            f"{label:<{col_w}}"
            f"{r['solve_rate']:>{num_w}.1f}"
            f"{first_str:>{num_w}}"
            f"{stable_str:>{num_w}}"
            f"{r['steps_solved']:>{num_w}.1f}"
            f"{r['pct_runs_stable']:>{num_w}.1f}"
        )

    print(divider)
    print("\n(Sorted by Stable Ep — lower = reached reliable solving sooner)\n")
    print("Columns:")
    print("  Solve Rate %   — % of all episodes that reached the goal")
    print("  First Solve Ep — median episode of first-ever solve across runs")
    print(f"  Stable Ep      — median episode of first {STABLE_WINDOW} consecutive solves")
    print("  Steps (solved) — mean steps taken on episodes that did succeed")
    print("  % Runs Stable  — % of runs that ever achieved stable solving\n")


if __name__ == "__main__":
    run_benchmark()