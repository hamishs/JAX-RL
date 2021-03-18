'''
A JAX implementation of the Deep Q-Network algorithm (DQN).
'''

import jax 
import jax.numpy as jnp 
import numpy as np 

import random 
import time 
from typing import NamedTuple

# experience replay buffer
class Transition(NamedTuple):
	s: list # state
	a: int # action
	r: float # reward
	d: bool # done
	s_next: list # next state

class ExperienceReplay:
	'''Experience replay buffer to store past experience.'''

	def __init__(self, buffer_size):
		'''
		buffer_size : int - max number of transitions to store.
		'''
		self.buffer_size = buffer_size 
		self.buffer = [] 
	
	def update(self, transition):
		''' Update the buffer with a transition.'''
		self.buffer.append(transition)
		if len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)
	
	def __len__(self):
		return len(self.buffer)
	
	def sample(self, n):
		''' Sample a batch of n transitions. Returns all as jax.numpy arrays.'''
		batch = Transition(*zip(*random.sample(self.buffer, n)))
		
		s = jnp.array(batch.s, dtype = jnp.float32)
		a = jnp.array(batch.a, dtype = jnp.int32)
		r = jnp.array(batch.r, dtype = jnp.float32)
		d = jnp.array(batch.d, dtype = jnp.float32)
		s_next = jnp.array(batch.s_next, dtype = jnp.float32)

		return s, a, r, d, s_next

# functions to create network and updates
def create_Q_network(h, key):
	''' Creates a MLP with network sizes h.
	h : list of int
	key : a jax.random.PRNGKey key.
	'''

	keys = jax.random.split(key, len(h))

	def init_layer(m, n, key, scale = 1e-2):
		w_key, b_key = jax.random.split(key)
		w = scale * jax.random.normal(w_key, (n, m))
		b = scale * jax.random.normal(b_key, (n,))
		return w, b
	
	return [init_layer(m, n, k) for m, n, k in zip(h[:-1], h[1:], keys)]

def forward(params, x):
	''' Forward function for a MLP with RELU connections and softmax output.
	params : list of tuples of jax arrays with parameters for each layer.'''

	for w, b in params[:-1]:
		x = jax.nn.relu(jnp.dot(w, x) + b)
	
	w, b = params[-1]
	return jnp.dot(w, x) + b

# create a batched version of the function
batch_forward = jax.vmap(forward, in_axes = (None, 0), out_axes = 0)

def target_loss(params, s, y, a):
	''' Computes the target loss for a given batch.
	params : list of tuples of jax arrays with parameters of the Q-network.
	s : jax.numpy array of states
	y : jax.numpy array of targets
	a : jax.numpy array of chosen actions.
	'''
	q = batch_forward(params, s)[jnp.arange(s.shape[0]), a]
	return jnp.mean((y - q) ** 2)

# apply jax.grad to get the gradient function
q_backward = jax.jit(jax.grad(target_loss))

def get_optimizer(lr = 1e-3):
	''' Returns a jax.jit function that applies one step of
	stochastic gradient descent to given parameters with
	given gradients using the learning rate.
	lr : float.
	'''

	def sgd(params, gradients):
		'''
		params : list of tuples of network parametesr.
		gradients : list of tuples of same shapes as params with gradients.
		'''
		new_params = []
		for (w, b), (dw, db) in zip(params, gradients):
			new_params.append((w - lr * dw, b - lr * db))
		return new_params
	
	return jax.jit(sgd)

# policies
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
			return int(jnp.argmax(forward(s)))

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

		self.t += 1

		T = self.T(self.t) if callable(self.T) else self.T

		prefs = forward(state)
		
		if exploration:
			prefs = jnp.exp(prefs / T)
			prefs /= prefs.sum()
			return int(jax.random.choice(key, n_actions, p = prefs))
		else:
			return int(jnp.argmax(prefs))

# agent
class Agent:

	def __init__(self, buffer_size, key, n_states, n_actions, hidden, policy, gamma, lr):

		self.buffer = ExperienceReplay(buffer_size)

		self.policy = policy

		self.n_states = n_states
		self.n_actions = n_actions
		
		self.gamma = gamma

		key, subkey = jax.random.split(key)
		self.params = create_Q_network([n_states] + hidden + [n_actions], subkey)
		self.target_params = self.params.copy()

		self.q_forward = batch_forward
		self.q_backward = q_backward

		self.optimizer = get_optimizer(lr = lr)

	def act(self, key, s, exploration = True):
		key, subkey = jax.random.split(key)

		s = jnp.array(s)[jnp.newaxis, :]
		forward = lambda x: self.q_forward(self.params, x).squeeze()

		return self.policy(subkey, s, self.n_actions, forward, exploration)

	
	def train(self, transition, batch_size):

		self.buffer.update(transition)

		if len(self.buffer) < batch_size:
			return self.params

		s, a, r, d, s_next = self.buffer.sample(batch_size) 

		q_next = self.q_forward(self.target_params, s_next)
		y = r + (1 - d) * self.gamma * jnp.max(q_next, axis = -1)
		
		gradients = self.q_backward(self.params, s, y, a)
		self.params = self.optimizer(self.params, gradients)

		return gradients
	
	def update_target(self):
		self.target_params = self.params.copy()
		
		