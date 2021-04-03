'''
Solve a simple grid world with tabular RL.
'''

import jax
import jax.numpy as jnp
import numpy as np 
import matplotlib.pyplot as plt

from jax_rl.tabular import QLearning, DoubleQLearning, SARSA, ExpectedSARSA
from jax_rl.policies import EpsilonGreedy
from jax_rl.environment import GridWorld

# setup gridworld

grid = np.array([
	[0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 0],
	[0, 1, 0, 1, 1, 1, 1, 1, 0],
	[0, 1, 0, 1, 0, 1, 1, 1, 0],
	[0, 1, 0, 1, 0, 1, 1, 1, 0],
	[0, 1, 1, 1, 0, 1, 1, 1, 0],
	[0, 0, 0, 0, 0, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 0, 0, 0 ,0]])

start = (1, 2)
end = (6, 7)
wall_reward = -10.0
step_reward = -0.1

class FlatWrapper:
	''' Wrapper to return integer states for grid world.'''

	def __init__(self, env, rows, cols):
		self.env = env 
		self.rows = rows 
		self.cols = cols

	def reset(self):
		s = self.env.reset()
		return s[0] * self.cols + s[1]

	def step(self, a):
		s_, r, d, info = self.env.step(a)
		s = s_[0] * self.cols + s_[1]
		return s, r, d, info 

env = FlatWrapper(GridWorld(grid, start, end, wall_reward, step_reward),
	grid.shape[0], grid.shape[1])

# train each algorithm on the grid problem

q_learning = QLearning(4, grid.size, 4, 0.99, EpsilonGreedy(0.1), 0.1)
q_rs = q_learning.train_on_env(env, 30, verbose = 5)

double_q = DoubleQLearning(5, grid.size, 4, 0.99, EpsilonGreedy(0.1), 0.1)
qd_rs = double_q.train_on_env(env, 30, verbose = 5)

sarsa = SARSA(6, grid.size, 4, 0.99, EpsilonGreedy(0.1), 0.1)
s_rs = sarsa.train_on_env(env, 30, verbose = 5)

expected_sarsa = ExpectedSARSA(7, grid.size, 4, 0.99, EpsilonGreedy(0.1), 0.1)
es_rs = expected_sarsa.train_on_env(env, 30, verbose = 5)

# plot results
plt.plot(q_rs, label = 'Q-learning')
plt.plot(qd_rs, label = 'Double Q-learning')
plt.plot(s_rs, label = 'SARSA')
plt.plot(es_rs, label = 'Expected SARSA')
plt.legend()
plt.show()








