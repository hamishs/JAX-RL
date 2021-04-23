import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

from functools import partial

from jax_rl.algorithms import BaseAgent
from jax_rl.buffer import EpisodicBuffer

class DRQN(BaseAgent):
	''' Deep Recurrent Q-network using double Q-learning.'''

	def __init__(self, key, n_states, n_actions, gamma, buffer_size,
		max_len, policy, model, init_state, lr):
		'''
		model must take sequential inputs and a hidden state.
		init_state must provide the initial state for a given batch_size.
		key : int for seeding
		n_states : int - dimensionality of state space
		n_actions : int - dimensionality of action space
		gamma : float - discount rate
		buffer_size : int - max number of episodes to store
		max_len : int - max length of each episode to store
		policy : jax_rl.policy.Policy
		model : hk.transform - taking sequential inputs and hidden state
		init_state : function to return initial hidden state given batch size
		lr : float - learning rate
		'''
		super(DRQN, self).__init__(key, n_states, n_actions, gamma)

		self.buffer = EpisodicBuffer(buffer_size, max_len = max_len)
		self.policy = policy

		# Q-network and parameters
		self.params = model.init(next(self.prng), jnp.ones((1, 5, n_states)))
		self.update_target()
		self.init_state = init_state
		self.q_network = model.apply

		# optimiser
		self.opt_update, self.opt_state = self.init_optimiser(lr, self.params)

	def act(self, s, exploration = True):
		''' Get an action from the q-network given the state.
		s - torch.FloatTensor shape (1, 1, n_states) current state
		exploration - bool - wether to choose greedy action or use epsilon greedy.
		Returns : action - int
		'''
		assert s.shape == (1, 1, self.n_states)
		q_values, self.hidden_state = self.q_network(self.params, s, self.hidden_state)

		return self.policy(next(self.prng), q_values, exploration)

	def train(self):
		''' Train the agent on a single episode. Uses the double q-learning target.
		Returns: td loss - float.'''
		if len(self.buffer.buffer) > 2:
			s, a, r, d = self.buffer.sample()

			s = jnp.array([s]) # (1, ep_length, n_states)
			a = jnp.array(a) # (ep_length)
			r = jnp.array(r) # (ep_length)
			d = jnp.array(d) # (ep_length)

			# compute gradients
			loss, gradients = jax.value_and_grad(self.loss_fn)(
				self.params, self.target_params, s, a, r, d)

			# apply gradients
			updates, self.opt_state = self.opt_update(gradients, self.opt_state, self.params)
			self.params = optax.apply_updates(self.params, updates)

			return loss

	def start_episode(self):
		''' Tell the buffer to start a new episode and reset the hidden state of the RNN.'''
		self.buffer.start_episode()
		self.hidden_state = self.init_state(1)

	def init_optimiser(self, lr, params):
		opt_init, opt_update = optax.adam(lr)
		opt_state = opt_init(params)
		return opt_update, opt_state

	@partial(jax.jit, static_argnums = 0)
	def loss_fn(self, params, target_params, s, a, r, d):
		# td targets
		h = self.init_state(1)
		a_targ = jnp.argmax(self.q_network(params, s[:,1:], h)[0], axis = -1).squeeze(0) # (ep_length - 1)
		q_next = self.q_network(target_params, s[:,1:], h)[0].squeeze(0) # (ep_length - 1, 2)
		q_next = jnp.hstack((q_next[jnp.arange(a_targ.shape[0]), a_targ], jnp.zeros((1,)))) # (ep_length,)
		td_target = r + self.gamma * jax.lax.stop_gradient(q_next) * (1.0 - d) # (ep_length,)

		# q-values
		q_values = self.q_network(params, s, h)[0].squeeze(0)
		q_values = q_values[jnp.arange(a.shape[0]), a]

		return jnp.square((q_values - td_target)).mean()

	def update_target(self):
		''' Update the weights of the target network.'''
		self.target_params = hk.data_structures.to_immutable_dict(self.params)

	def update_buffer(self, s, a, r, d):
		''' Update the buffer with a transition.'''
		self.buffer.update(s, a, r, d)

	def train_on_env(self, env, episodes, update_freq, verbose = None):
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

			self.start_episode()
			s = env.reset()
			d = False
			ep_reward = 0.0
			while not d:
				a = self.act(jnp.array(s)[jnp.newaxis, jnp.newaxis, :], exploration = True)
				s_next, r, d, _ = env.step(a)
				ep_reward += r
				self.update_buffer(s, a, r, d)
				s = s_next

			# ep end
			ep_rewards.append(ep_reward)
			if episode % update_freq == 0:
				loss = self.train()
				if loss is not None: losses.append(loss)

			if verbose is not None:
				if episode % verbose == 0:
					print('Episode {} Reward {:.4f} Loss {:.4f}'.format(episode,
						np.mean(ep_rewards[-verbose:]),
						np.mean(losses[-verbose:])))

		return ep_rewards, losses

