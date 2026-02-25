#algorithms.py


import numpy as np


class MonteCarlo:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.returns = {}

    def needs_full_episode(self):
        return True

    def reset(self):
        self.returns = {}

    def learn(self, Q, episode):
        G = 0
        visited = set()
        for i in range(len(episode) - 1, -1, -1):
            (x, y), a, r = episode[i]
            G = r + self.gamma * G
            if ((x, y), a) not in visited:
                visited.add(((x, y), a))
                key = (x, y, a)
                if key not in self.returns:
                    self.returns[key] = []
                self.returns[key].append(G)
                Q[(x, y)][a] = np.mean(self.returns[key])


#TEMPORAL DIFFERENCE (TD) METHODS
class SARSA:    #on-policy
    def __init__(self, gamma=0.99, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha

    def needs_full_episode(self):
        return False

    def reset(self):
        pass  # no history to clear — learns in-place

    def learn(self, Q, s, a, r, s_next, a_next):
        """Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))"""
        q_current = Q[s][a]
        q_next = Q[s_next][a_next] if a_next else 0.0
        Q[s][a] += self.alpha * (r + self.gamma * q_next - q_current)


class QLearning:    #off-policy
    def __init__(self, gamma=0.99, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha

    def needs_full_episode(self):
        return False

    def reset(self):
        pass

    def learn(self, Q, s, a, r, s_next, a_next=None):
        """Q(s,a) += alpha * (r + gamma * max(Q(s')) - Q(s,a))"""
        max_q_next = max(Q[s_next].values())
        Q[s][a] += self.alpha * (r + self.gamma * max_q_next - Q[s][a])