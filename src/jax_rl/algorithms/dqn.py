import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

from functools import partial

from jax_rl.algorithms import BaseAgent
from jax_rl.buffer import ExperienceReplay
from jax_rl.utils import Transition

class DQN(BaseAgent):
	''' Deep Q-network'''

	def __init__(self, key, n_states, n_actions, gamma, buffer_size, policy, model, lr):
		'''
		model must take sequential inputs and a hidden state.
		init_state must provide the initial state for a given batch_size.
		'''
		super(DQN, self).__init__(key, n_states, n_actions, gamma)

		self.buffer = ExperienceReplay(buffer_size)
		self.policy = policy

		# Q-network and parameters
		self.params = model.init(next(self.prng), jnp.ones((1, n_states)))
		self.update_target()
		self.q_network = model.apply

		# optimiser
		self.opt_update, self.opt_state = self.init_optimiser(lr, self.params)

	def act(self, s, exploration = True):
		''' Get an action from the q-network given the state.
		s - torch.FloatTensor shape (1, 1, n_states) current state
		exploration - bool - wether to choose greedy action or use epsilon greedy.
		Returns : action - int
		'''
		assert s.shape == (1, self.n_states)
		q_values = self.q_network(self.params, s)

		return self.policy(next(self.prng), q_values, exploration)

	def train(self, batch_size):
		''' Train the agent on a single episode. Uses the double q-learning target.
		Returns: td loss - float.'''
		if len(self.buffer.buffer) > batch_size:
			s, a, r, d, s_next = self.buffer.sample(batch_size)

			# compute gradients
			loss, gradients = jax.value_and_grad(self.loss_fn)(
				self.params, self.target_params, s, a, r, d, s_next)

			# apply gradients
			updates, self.opt_state = self.opt_update(gradients, self.opt_state, self.params)
			self.params = optax.apply_updates(self.params, updates)

			return loss

	def init_optimiser(self, lr, params):
		opt_init, opt_update = optax.adam(lr)
		opt_state = opt_init(params)
		return opt_update, opt_state

	@partial(jax.jit, static_argnums = 0)
	def loss_fn(self, params, target_params, s, a, r, d, s_next):
		# td targets
		y = r + self.gamma * self.q_network(target_params, s_next).max(-1) * (1.0 - d)

		# q-values
		q_values = self.q_network(params, s)
		q_values = q_values[jnp.arange(a.shape[0]), a]

		return jnp.square((q_values - y)).mean()

	def update_target(self):
		''' Update the weights of the target network.'''
		self.target_params = hk.data_structures.to_immutable_dict(self.params)

	def update_buffer(self, transition):
		''' Update the buffer with a transition.'''
		self.buffer.update(transition)

	def train_on_env(self, env, episodes, batch_size, target_freq, verbose = None):
		''' Train on a given environment.
		env : environment with methods reset and step e.g. gym CartPole-v1
		episodes : number of training episodes
		update_freq : frequency of training episodes
		train_steps : number of gradient descent steps each training epsiode
		verbose : wether to print current rewards. Given as int refering to print frequency.
		'''
		ep_rewards = []
		losses = []
		for episode in range(episodes):
			s = env.reset()
			d = False
			ep_reward = 0.0
			while not d:
				a = self.act(jnp.array(s)[jnp.newaxis, :], exploration = True)
				s_next, r, d, _ = env.step(a)
				ep_reward += r
				self.update_buffer(Transition(s=s, a=a, r=r, d=d, s_next=s_next))
				loss = self.train(batch_size)
				if loss is not None: losses.append(loss)
				s = s_next

			# ep end
			ep_rewards.append(ep_reward)

			if episode % target_freq == 0:
				self.update_target()

			if verbose is not None:
				if episode % verbose == 0:
					print('Episode {} Reward {:.4f} Loss {:.4f}'.format(episode,
						np.mean(ep_rewards[-verbose:]),
						np.mean(losses[-verbose:]))) # need to * by ep-length here

		return ep_rewards, losses

class DDQN(DQN):
	''' DQN with double Q-learning. Overwrites the loss function with the DDQN target.'''

	def __init__(self, *args):
		super(DDQN, self).__init__(*args)

	@partial(jax.jit, static_argnums = 0)
	def loss_fn(self, params, target_params, s, a, r, d, s_next):
		# td targets
		a_targ = jnp.argmax(jax.lax.stop_gradient(self.q_network(params, s_next)), axis = -1)
		q_targ = self.q_network(target_params, s_next)[jnp.arange(a_targ.shape[0]), a_targ]
		y = r + self.gamma * q_targ * (1.0 - d)

		# q-values
		q_values = self.q_network(params, s)
		q_values = q_values[jnp.arange(a.shape[0]), a]

		return jnp.square((q_values - y)).mean()
