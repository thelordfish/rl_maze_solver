import numpy as np
import random
import matplotlib.pyplot as plt
from maze_gen import generate_prim_maze
from mdp_env import ACTIONS, GridMazeMDP
from viz_turtle import TurtleMazeViz
from policy_visualiser import PolicyVisualiser

# --- Configuration ---
WIDTH, HEIGHT = 6, 6
EPISODES = 100
GAMMA = 0.99
TAU = 1.0 #temperature - i.e. strength of the softmax
STEP_CAP = 500

def get_policy_map(Q, tau):
    """Helper to format Q-table for the Visualiser."""
    policy = {}
    for x in range(WIDTH):
        for y in range(HEIGHT):
            q_s = Q[(x, y)]
            max_q = max(q_s.values())
            exp_q = {a: np.exp((q_s[a] - max_q) / tau) for a in ACTIONS}
            z = sum(exp_q.values())
            policy[(x, y)] = {a: exp_q[a] / z for a in ACTIONS}
    return policy

def run_sync_mc():
    # 1. Environment & Turtle Setup
    maze = generate_prim_maze(WIDTH, HEIGHT, seed=42) #
    mdp = GridMazeMDP(WIDTH, HEIGHT, maze.walls) #
    
    # Initialize both Visualisers
    turtle_viz = TurtleMazeViz(WIDTH, HEIGHT, maze.walls) #
    policy_viz = PolicyVisualiser(WIDTH, HEIGHT)
    
    # 2. RL Data
    Q = {(x, y): {a: 0.0 for a in ACTIONS} for x in range(WIDTH) for y in range(HEIGHT)}
    returns = {(x, y, a): [] for x in range(WIDTH) for y in range(HEIGHT) for a in ACTIONS}

    plt.ion() # Turn on Matplotlib interactive mode

    for ep in range(EPISODES):
        episode_data = []
        s = mdp.start() #
        
        # --- A. Episode Loop (Turtle) ---
        while not mdp.is_terminal(s): #
            q_s = Q[(s.x, s.y)]
            # Softmax selection
            max_q = max(q_s.values())
            probs = [np.exp((q_s[a] - max_q) / TAU) for a in ACTIONS]
            probs = [p / sum(probs) for p in probs]
            
            a = random.choices(ACTIONS, weights=probs)[0]
            s_next = mdp.transition(s, a) #
            r = mdp.reward(s, a, s_next) #
            
            episode_data.append(((s.x, s.y), a, r))
            s = s_next
            
            # Update Turtle Window
            turtle_viz.move_agent(s, delay=0.001) #
            if len(episode_data) > STEP_CAP: break

        # --- B. Learning (Monte Carlo) ---
        G = 0
        visited = set()
        for i in range(len(episode_data)-1, -1, -1):
            (x, y), a, r = episode_data[i]
            G = r + GAMMA * G
            if ((x, y), a) not in visited:
                visited.add(((x, y), a))
                returns[(x, y, a)].append(G)
                Q[(x, y)][a] = np.mean(returns[(x, y, a)])

        # --- C. Update Policy Visualiser (Matplotlib) ---
        current_policy = get_policy_map(Q, TAU)
        policy_viz.update(current_policy)
        
        # Sync Turtle Policy Overlay
        turtle_viz.draw_policy(Q) #
        
        # Essential: Let Matplotlib draw the frame
        plt.pause(0.01) 
        print(f"Episode {ep} completed.")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_sync_mc()