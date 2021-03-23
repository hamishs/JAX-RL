'''
Deep Q-Network (DQN) as presented in
Playing Atari with Deep Reinforcement Learning
(Mnih et al., 2013)
https://arxiv.org/abs/1312.5602

Double DQN (DDQN) as presented in
Deep Reinforcement Learning with Double Q-learning
(van Hasselt et al., 2015)
https://arxiv.org/abs/1509.06461
'''

import jax 
import jax.numpy as jnp 
import numpy as np
import haiku as hk
import optax

from utils import ExperienceReplay, Transition

class DQNAgent:
	''' A Deep Q-Network agent.'''

	def __init__(self, buffer_size, key, n_states, n_actions, model, policy, gamma, lr):
		'''
		buffer_size : int -maximum memory size
		key : jax.random.PRNGKey
		n_states : int - dimensionality of state space
		n_actions : int - dimensionality of action space
		model : the q-network taking states to action values
		policy : exploration policy
		gamma : float - discount for future rewards
		lr : float - learning rate of mlp
		'''

		self.buffer = ExperienceReplay(buffer_size)

		self.policy = policy

		self.n_states = n_states
		self.n_actions = n_actions
		
		self.gamma = gamma

		key, subkey = jax.random.split(key)
		self.params = model.init(subkey, jnp.ones([2, n_states]))
		self.target_params = hk.data_structures.to_immutable_dict(self.params)
		self.q_forward = model.apply # takes args : params, key, s

		def target_loss(params, key, s, y, a):
			''' Computes the target loss for a given batch.'''
			q = model.apply(params, key, s)[jnp.arange(s.shape[0]), a]
			return jnp.mean((y - q) ** 2)
		q_backward = jax.jit(jax.grad(target_loss))
		self.q_backward = q_backward

		opt_init, self.opt_update = optax.sgd(lr)
		self.opt_state = opt_init(self.params)

	def act(self, key, s, exploration = True):
		key, subkey = jax.random.split(key)

		s = jnp.array(s)[jnp.newaxis, :]
		forward = lambda x: self.q_forward(self.params, subkey, x).squeeze()

		return self.policy(subkey, s, self.n_actions, forward, exploration)
	
	def train(self, transition, batch_size, key):

		key, subkey = jax.random.split(key)

		self.buffer.update(transition)

		if len(self.buffer) >= batch_size:

			s, a, r, d, s_next = self.buffer.sample(batch_size) 

			q_next = self.q_forward(self.target_params, subkey, s_next)
			y = r + (1 - d) * self.gamma * jnp.max(q_next, axis = -1)
			
			# update parameters
			gradients = self.q_backward(self.params, subkey, s, y, a)
			updates, self.opt_state = self.opt_update(gradients, self.opt_state, self.params)
			self.params = optax.apply_updates(self.params, updates)
	
	def update_target(self, soft = False, tau  = 0.5):
		''' Updates the target network with parameters from the Q-network.
		soft : bool = False. Wether to perfrom a soft update
			(keep a percent of the target network parameters)
		tau : float = 0.5. If soft = True then the percentage of target network parameters to keep.
		'''
		if soft:
			self.target_params = jax.tree_util.tree_multimap(
				lambda x, y: tau * x + (1 - tau) * y,
				self.target_params,
				self.params)
		else:
			self.target_params = hk.data_structures.to_immutable_dict(self.params)

class DDQNAgent(DQNAgent):
	''' A Double-DQN agent (https://arxiv.org/pdf/1509.06461.pdf).'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def train(self, transition, batch_size, key = None):

		key, subkey = jax.random.split(key) if key is not None else None, None

		self.buffer.update(transition)

		if len(self.buffer) < batch_size:
			return self.params

		s, a, r, d, s_next = self.buffer.sample(batch_size) 

		q_actions = jnp.argmax(self.q_forward(self.params, subkey, s_next), axis = -1)
		y = r + (1 - d) * self.gamma * self.q_forward(self.target_params, subkey, s_next)[jnp.arange(s_next.shape[0]), q_actions]
		
		# update parameters
		gradients = self.q_backward(self.params, subkey, s, y, a)
		updates, self.opt_state = self.opt_update(gradients, self.opt_state, self.params)
		self.params = optax.apply_updates(self.params, updates)


if __name__ == '__main__':

	# implements the DQN agent on CartPole-v0.

	import gym
	import matplotlib.pyplot as plt
	import argparse

	from policies import EpsilonGreedy
	from models import mlp
	 
	parser = argparse.ArgumentParser(description = 'Run DQN on CartPole-v0')
	parser.add_argument('-agent', type = str, default = 'dqn')
	parser.add_argument('-eps_min', type = float, default = 0.01)
	parser.add_argument('-eps_max', type = float, default = 0.15)
	parser.add_argument('-eps_decay', type = float, default = 1e-5)
	parser.add_argument('-buffer_size', type = int, default = 5000)
	parser.add_argument('-gamma', type = float, default = 0.97)
	parser.add_argument('-lr', type = float, default = 1e-3)
	parser.add_argument('-episodes', type = int, default = 1500)
	parser.add_argument('-batch_size', type = int, default = 32)

	args = parser.parse_args()

	Agent = DDQNAgent if args.agent == 'ddqn' else DQNAgent
	epsilon_min = args.eps_min
	epsilon_max = args.eps_max
	epsilon_decay = args.eps_decay
	buffer_size = args.buffer_size
	gamma = args.gamma
	lr = args.lr
	episodes = args.episodes
	batch_size = args.batch_size

	env = gym.make('CartPole-v0')

	key = jax.random.PRNGKey(0)
	policy = EpsilonGreedy(lambda t: epsilon_min + epsilon_max * jnp.exp(-t * epsilon_decay))
	model = mlp.create_mlp([32, 32, 2])
	agent = Agent(buffer_size, key, 4, 2, model, policy, gamma, lr)

	ep_rewards = []
	for episode in range(episodes):

		s = env.reset()
		d = False
		ep_reward = 0.0

		while not d:

			key, subkey = jax.random.split(key)
			a = agent.act(subkey, s)

			s_next, r, d, _ = env.step(a)
			ep_reward += r
			transition = Transition(s = s, a = a, r = r, d = d, s_next = s_next)

			agent.train(transition, batch_size, subkey)

			s = s_next

		ep_rewards.append(ep_reward)

		if episode % 50 == 0:
			agent.update_target()
			print('Episode {} Mean reward {}'.format(episode, np.mean(ep_rewards[-50:])))

	plt.plot(ep_rewards)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Episodic reward for {} on CartPole-v0.'.format(args.agent.upper()))
	plt.show()
