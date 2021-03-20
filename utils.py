import jax
import jax.numpy as jnp 
import numpy as np 

from typing import NamedTuple
import random

class Transition(NamedTuple):
	s: list # state
	a: int # action
	r: float # reward
	d: bool # done
	s_next: list # next state

class ExperienceReplay:
	'''Experience replay buffer to store past experience.'''

	def __init__(self, buffer_size, action_dtype = jnp.int32):
		'''
		buffer_size : int - max number of transitions to store.
		'''
		self.buffer_size = buffer_size 
		self.buffer = [] 

		self.action_dtype = action_dtype
	
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
		a = jnp.array(batch.a, dtype = self.action_dtype)
		r = jnp.array(batch.r, dtype = jnp.float32)
		d = jnp.array(batch.d, dtype = jnp.float32)
		s_next = jnp.array(batch.s_next, dtype = jnp.float32)

		return s, a, r, d, s_next

class GaussianNoise:

		def __init__(self, mean = 0.0, sd = 1.0):

			self.mean = mean
			self.sd = sd
			self.t = 0

		def __call__(self, key):
			key, subkey = jax.random.split(key)

			mu = self.mean(self.t) if callable(self.mean) else self.mean
			sigma = self.sd(self.t) if callable(self.sd) else self.sd 

			self.t += 1

			return (jax.random.normal(subkey) + mu) * sigma