import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.k = 1
        self.memory = []
        if mode == "mc_control":
            self.alpha = 0.02
            self.gamma = 0.9
        elif mode == "q_learning":
            self.alpha = 0.2
            self.gamma = 0.8
        else:
            self.alpha = 0.000021
            self.gamma = 0.9


    def select_action(self, state):
        if self.mode == "test_mode":
            epsilon = 0

        elif self.mode == "mc_control":
            if self.k < 400:
                epsilon = 1.0 / ((self.k // 100) + 1)
            elif self.k < 23000:
                epsilon = 0.2
            else:
                epsilon = 1.0 / ((self.k // 100) + 1)

        else:
            epsilon = 1/self.k

        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        if self.mode == "q_learning":
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
            if done:
                self.k += 1

        else:
            if done:
                G_t = 0
                for sample in reversed(self.memory):
                    state, action, reward = sample
                    G_t = self.gamma * G_t + reward
                    self.Q[state][action] += self.alpha * (G_t - self.Q[state][action])
                self.memory.clear()
                self.k += 1

            else:
                self.memory.append([state, action, reward])

