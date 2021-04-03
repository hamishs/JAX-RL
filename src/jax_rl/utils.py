import jax
import jax.numpy as jnp 
import numpy as np
import haiku as hk 

from typing import NamedTuple

class Transition(NamedTuple):
	s: list # state
	a: int # action
	r: float # reward
	d: bool # done
	s_next: list # next state

def entropy(dist, axis = -1):
	return -(jnp.log(dist) * dist).sum(axis = axis)

def lstm_initial_state(units, batch_size = None):
	''' Computes the intial hidden state of an LSTM cell using
	given unit size and batch size.'''
	state = hk.LSTMState(hidden=jnp.zeros([units]), cell=jnp.zeros([units]))
	if batch_size is not None:
		broadcast = lambda x : jnp.broadcast_to(x, (batch_size,) + x.shape)
		state = jax.tree_map(broadcast, state)
	return state

class GaussianNoise:

	def __init__(self, mean = 0.0, sd = 1.0):

		self.mean = mean
		self.sd = sd
		self.t = 0

	def __call__(self, key):
		mu = self.mean(self.t) if callable(self.mean) else self.mean
		sigma = self.sd(self.t) if callable(self.sd) else self.sd 

		self.t += 1

		return (jax.random.normal(key) + mu) * sigma
