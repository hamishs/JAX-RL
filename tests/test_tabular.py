import numpy as np

from jax_rl.environment import GridWorld
from jax_rl.policies import EpsilonGreedy
from jax_rl.tabular import SARSA, DoubleQLearning, ExpectedSARSA, QLearning

# setup gridworld for testing

grid = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
)

start = (1, 2)
end = (4, 5)
wall_reward = -10.0
step_reward = -0.1


class FlatWrapper:
    """Wrapper to return integer states for grid world."""

    def __init__(self, env, rows, cols):
        self.env = env
        self.rows = rows
        self.cols = cols

    def reset(self):
        s, _ = self.env.reset()
        return s[0] * self.cols + s[1]

    def step(self, a):
        s_, r, d, info = self.env.step(a)
        s = s_[0] * self.cols + s_[1]
        return s, r, d, info


env = FlatWrapper(
    GridWorld(grid, start, end, wall_reward, step_reward), grid.shape[0], grid.shape[1]
)

# test algos

q_learning = QLearning(4, grid.size, 4, 0.99, EpsilonGreedy(0.1), 0.1)
print(q_learning.train_on_env(env, 3, verbose=1))

double_q = DoubleQLearning(5, grid.size, 4, 0.99, EpsilonGreedy(0.1), 0.1)
print(double_q.train_on_env(env, 3, verbose=1))

sarsa = SARSA(6, grid.size, 4, 0.99, EpsilonGreedy(0.1), 0.1)
print(sarsa.train_on_env(env, 3, verbose=1))

expected_sarsa = ExpectedSARSA(7, grid.size, 4, 0.99, EpsilonGreedy(0.1), 0.1)
print(expected_sarsa.train_on_env(env, 3, verbose=1))
