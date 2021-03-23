import jax 
import jax.numpy as jnp 
import numpy as np 

class EpsilonGreedy:
	''' Greedy/random policy with probability epsilon/1-epsilon.'''

	def __init__(self, epsilon):
		''' epsilon : float or callable giving exploration rate.'''

		self.epsilon = epsilon 
		self.t = 0 # step counter

	def __call__(self, key, state, n_actions, forward, exploration = True):

		self.t += 1

		eps = self.epsilon(self.t) if callable(self.epsilon) else self.epsilon
		
		if exploration and (jax.random.uniform(key, shape = (1,))[0] > 1 - eps):
			return int(jax.random.randint(key, shape = (1,), minval = 0, maxval = n_actions))
		else:
			return int(jnp.argmax(forward(state)))

class BoltzmannPolicy:
	''' Exploration with a Boltzmann distribution over
	the Q-values with temperature.'''

	def __init__(self, T):
		'''
		T : float or callable giving temperature parameter.
		'''

		self.T = T 
		self.t = 0 # step counter

	def __call__(self, key, state, n_actions, forward, exploration = True):
		'''
		key : jax.random.PRNGKey
		state : current state as a jnp array
		n_actions : int - dimensionality of action space
		forward : function taking state to action preferences
		exploration : bool - wether to include exploration.
		'''

		self.t += 1
		T = self.T(self.t) if callable(self.T) else self.T
		prefs = jax.nn.softmax(forward(state))
		
		if exploration:
			return int(jax.random.choice(key, n_actions, p = prefs))
		else:
			return int(jnp.argmax(prefs))
				