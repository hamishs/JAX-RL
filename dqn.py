import jax 
import jax.numpy as jnp 
import numpy as np

from models import mlp
from utils import Transition, ExperienceReplay

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

class Agent:

	def __init__(self, buffer_size, key, n_states, n_actions, hidden, policy, gamma, lr):

		self.buffer = ExperienceReplay(buffer_size)

		self.policy = policy

		self.n_states = n_states
		self.n_actions = n_actions
		
		self.gamma = gamma

		key, subkey = jax.random.split(key)
		self.params = mlp.mlp_init_fn([n_states] + hidden + [n_actions], subkey)
		self.target_params = self.params.copy()

		self.q_forward = mlp.batch_forward
		self.q_backward = q_backward

		self.optimizer = mlp.get_optimizer(lr = lr)

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
		
		