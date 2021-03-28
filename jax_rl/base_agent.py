from abc import ABC, abstractmethod
import jax
import haiku as hk

class BaseAgent(ABC):

	def __init__(self, key, n_states, n_actions, gamma):
		self.prng = hk.PRNGSequence(key)
		self.n_states = n_states
		self.n_actions = n_actions
		self.gamma = gamma

	@abstractmethod
	def train(self):
		pass

	@abstractmethod
	def act(self, s):
		pass