import jax
import jax.numpy as jnp 
import numpy as np 
import random
from abc import ABC, abstractmethod
from collections import deque

from jax_rl.utils import Transition

class Buffer(ABC):

	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.reset()

	@abstractmethod
	def update(self, transition):
		pass

	@abstractmethod
	def __len__(self):
		pass

	@abstractmethod
	def reset(self):
		pass

class ExperienceReplay(Buffer):
	'''Experience replay buffer to store past experience.'''

	def __init__(self, buffer_size, action_dtype = jnp.int32):
		'''
		buffer_size : int - max number of transitions to store.
		'''
		super(ExperienceReplay, self).__init__(buffer_size)
		self.action_dtype = action_dtype
	
	def update(self, transition):
		''' Update the buffer with a transition.'''
		self.buffer.append(transition)
		if len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)
	
	def __len__(self):
		return len(self.buffer)

	def reset(self):
		self.buffer = []
	
	def sample(self, n):
		''' Sample a batch of n transitions. Returns all as jax.numpy arrays.'''
		batch = Transition(*zip(*random.sample(self.buffer, n)))
		
		s = jnp.array(batch.s, dtype = jnp.float32)
		a = jnp.array(batch.a, dtype = self.action_dtype)
		r = jnp.array(batch.r, dtype = jnp.float32)
		d = jnp.array(batch.d, dtype = jnp.float32)
		s_next = jnp.array(batch.s_next, dtype = jnp.float32)

		return s, a, r, d, s_next

class TrajectoryBuffer(Buffer):
	''' Buffer to store trajectories in order and store advantages.'''
	
	def __init__(self, buffer_size):
		super(TrajectoryBuffer, self).__init__(buffer_size)
		
	def reset(self):
		self.state = []
		self.action = []
		self.reward = []
		self.done = []
		self.state_next = []
		self.probs = []
		self.advs = []
		self.tds = jnp.array([])
		self.n = 0

	def __len__(self):
		return len(self.state)
	
	def update(self, s, a, r, d, s_next, pi):
		self.state.append(s)
		self.action.append([a])
		self.reward.append([r])
		self.done.append([d])
		self.state_next.append(s_next)
		self.probs.append(pi)
		
		if self.n < self.buffer_size:
			self.n += 1
		else:
			self.state.pop(0)
			self.action.pop(0)
			self.reward.pop(0)
			self.done.pop(0)
			self.state_next.pop(0)
			self.probs.pop(0)
			self.advs.pop(0)
			self.tds = self.tds[1:]
	
	@property
	def data(self):
		''' Return memory as arrays.'''
		s = jnp.array(self.state)
		a = jnp.array(self.action)
		r = jnp.array(self.reward)
		d = jnp.array(self.done)
		s_next = jnp.array(self.state_next)
		pi = jnp.array(self.probs)
		advs = jnp.array(self.advs)
		return s, a, r, d, s_next, pi, advs, self.tds

class EpisodicBuffer(Buffer):
	'''Buffer to store full episodes at a time.'''

	def __init__(self, buffer_size, max_len = 50):
		super(EpisodicBuffer, self).__init__(buffer_size)
		self.max_len = max_len
		
	def reset(self):
		self.buffer = deque(maxlen = self.buffer_size)

	def __len__(self):
		return len(self.buffer)

	def start_episode(self):
		self.buffer.append([])

	def update(self, s, a, r, d):
		n = len(self.buffer) - 1
		if len(self.buffer[n]) < self.max_len:
			self.buffer[n].append([s, a, r, d])

	def sample(self):
		idx = np.random.randint(0, len(self.buffer)-2) # exclude incomplete
		batch = self.buffer[idx] 
		return tuple(zip(*batch))

if __name__ == '__main__':

	_buffer = ExperienceReplay(100)
	_buffer = TrajectoryBuffer(100)
	_buffer = EpisodicBuffer(100)

	print(len(_buffer))
