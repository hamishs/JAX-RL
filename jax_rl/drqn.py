import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

from functools import partial

from base_agent import BaseAgent
from buffer import EpisodicBuffer

class DRQN(BaseAgent):
	''' Deep Recurrent Q-network using double Q-learning.'''

	def __init__(self, key, n_states, n_actions, gamma, buffer_size, max_len, policy, model, init_state, lr):
		'''
		model must take sequential inputs and a hidden state.
		init_state must provide the initial state for a given batch_size.
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

		return self.policy(next(self.prng), self.n_actions, q_values, exploration)

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



if __name__ == '__main__':

	from policies import EpsilonGreedy
	from utils import lstm_initial_state

	import gym
	env = gym.make('CartPole-v0')

	# Example LSTM network
	def forward(s, hidden = None):
		'''
		Apply the LSTM over the input sequence with given initial state.
		s : (batch, seq_len, features)
		hidden : LSTM state (h, c).
		'''

		# extract features
		mlp1 = hk.nets.MLP([16, 16])
		s = hk.BatchApply(mlp1)(s) # (batch, seq_len, hidden_features)

		# LSTM
		lstm = hk.LSTM(32)
		if hidden is None: hidden = lstm.initial_state(s.shape[0])
		s, hidden = hk.dynamic_unroll(lstm, jnp.transpose(s, (1, 0, 2)), hidden)

		# output fully connected
		mlp2 = hk.nets.MLP([16, 1])
		s = hk.BatchApply(mlp2)(jnp.transpose(s, (1, 0, 2)))

		# s : (batch, seq_len, 1)
		# hidden = (h, c)
		# h : ()

		return s, hidden

	model = hk.without_apply_rng(hk.transform(forward))
	init_state = lambda batch_size: lstm_initial_state(32, batch_size = batch_size)

	policy = EpsilonGreedy(0.1)

	drqn = DRQN(0, 4, 2, 0.99, 1000, 200, policy, model, init_state, 1e-5)
	drqn.train_on_env(env, 500, 1, verbose = 10)

