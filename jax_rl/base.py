from abc import ABC, abstractmethod

import jax 
import jax.numpy as jnp 
import numpy as np
import haiku as hk

class JAXRLBase(ABC):
	'''
	Abstract base class for deep RL algorithms. All algorithms need an act
	method (to take the state and choose an action) and a train method (to
	learn from experiences).
	'''

	def __init__(self, key_seed, n_states, n_actions, gamma):
		'''
		key_seed : initialises a PRNGSequence.
		n_states : the dimensionality of the state space
		n_actions : the dimensionality of the action space
		gamma : the discount factor for the MDP.
		'''
		self.n_states = n_states
		self.n_actions = n_actions
		self.gamma = gamma
		self.prng = hk.PRNGSequence(key_seed)

	@abstractmethod
	def act(self, s):
		pass

	@abstractmethod
	def train(self):
		pass

