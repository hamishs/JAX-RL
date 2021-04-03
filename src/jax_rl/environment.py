import numpy as np
from abc import ABC, abstractmethod

class Environment(ABC):
	'''
	An environment for a Markov Decision Process
	Uses similar syntax to env from gym (could subclass instead).
	'''
	def __init__(self):
		pass

	@abstractmethod
	def reset(self):
		pass

	@abstractmethod
	def step(self, a):
		pass

class GridWorld(Environment):
	'''
	Discrete grid world for tabular algorithms. Grid is represented
	by a np array indicating the walls and valid squares. Actions are
	movements up, down, left or right. There is a start square and a
	single end square which is the only terminal state.

	Actions:
	0 = up
	1 = down
	2 = left
	3 = right
	'''
	def __init__(self, grid, start, end, wall_reward, step_reward, max_steps = 100):
		'''
		grid : 2D binary numpy array with 0 for wall and 1 for valid square.
		start : tuple indicating row and column of starting square.
		end : tuple indicating row and column of end square.
		wall_reward : the reward of bumping into a wall square.
		step_reward : the reward for each step taken in the environment.
		'''
		self.grid = grid 
		self.start = start 
		self.end = end
		self.wall_reward = wall_reward
		self.step_reward = step_reward
		self.max_steps = max_steps

		self.n_rows, self.n_cols = self.grid.shape

	def reset(self):
		self.state = self.start
		self.done = False
		self._t = 0
		return list(self.state)

	def step(self, a):

		if self.done:
			print('Done')
			raise ValueError

		row, col = self.state
		reward = self.step_reward
		info = ''

		# process action
		if a == 0:
			r_next, c_next = row + 1, col
		elif a == 1:
			r_next, c_next = row - 1, col
		elif a == 2:
			r_next, c_next = row, col - 1
		elif a == 3:
			r_next, c_next = row, col + 1
		else:
			print('Invalid action')
			raise ValueError

		# make sure in grid
		if (r_next < 0) or (r_next >= self.n_rows):
			r_next = row
			reward += self.wall_reward
			info = 'Tried to leave the grid'
		elif (r_next < 0) or (r_next >= self.n_rows):
			c_next = col 
			reward += self.wall_reward
			info = 'Tried to leave the grid'

		# if a wall then stay still
		if self.grid[r_next, c_next] == 0:
			r_next = row
			c_next = col
			reward += self.wall_reward
			info = 'Hit a wall'

		self.state = (r_next, c_next)

		# check if done
		self._t += 1
		if (self.state == self.end) or (self._t >= self.max_steps):
			self.done = True

		return list(self.state), reward, self.done, info



