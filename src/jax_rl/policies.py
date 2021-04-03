import jax
import jax.numpy as jnp 
import numpy as np
from abc import ABC, abstractmethod

class Policy(ABC):
	''' Policy base class.'''

	def __init__(self):
		pass

	def __call__(self, *args, **kwargs):
		return self.call(*args, **kwargs)

	@abstractmethod
	def call(self):
		pass

class EpsilonGreedy(Policy):
	''' Greedy/random policy with probability epsilon/1-epsilon.'''

	def __init__(self, epsilon):
		super(EpsilonGreedy, self).__init__()
		''' epsilon : float or callable giving exploration rate.'''
		self.epsilon = epsilon 
		self.t = 0 # step counter

	def call(self, key, n_actions, q_values, exploration = True):
		'''
		key : jax.random.PRNGKey.
		state : jnp.array (1, n_states) - current state.
		n_actions : int - number of actions.
		q_values : (n_actions) - estimated q-values for the current state.
		exploration : bool = True - wether to allow exploration or to be greedy.
		'''

		self.t += 1
		eps = self.epsilon(self.t) if callable(self.epsilon) else self.epsilon

		if exploration and (jax.random.uniform(key, shape = (1,))[0] > 1 - eps):
			return int(jax.random.randint(key, shape = (1,), minval = 0, maxval = n_actions))
		else:
			return int(jnp.argmax(q_values))

class BoltzmannPolicy(Policy):
	''' Exploration with a Boltzmann distribution over the Q-values with temperature.'''

	def __init__(self, T):
		'''
		T : float or callable giving temperature parameter.
		'''
		super(BoltzmannPolicy, self).__init__()
		self.T = T 
		self.t = 0 # step counter

	def call(self, key, n_actions, q_values, exploration = True):
		'''
		key : jax.random.PRNGKey.
		state : jnp.array (1, n_states) - current state.
		n_actions : int - number of actions.
		q_values : (n_actions) - estimated q-values for the current state.
		exploration : bool = True - wether to allow exploration or to be greedy.
		'''
		self.t += 1
		T = self.T(self.t) if callable(self.T) else self.T

		if exploration:
			prefs = jnp.exp(q_values / T)
			prefs /= prefs.sum()
			return int(jax.random.choice(key, n_actions, p = prefs))
		else:
			return int(jnp.argmax(q_values))

if __name__ == '__main__':

	eps = EpsilonGreedy(0.1)
	boltz = BoltzmannPolicy(1.0)