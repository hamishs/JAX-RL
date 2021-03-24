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

from functools import partial

from base import JAXRLBase
from utils import ExperienceReplay, Transition

class DQNAgent(JAXRLBase):
	''' A Deep Q-Network agent.'''

	def __init__(self, buffer_size, keyseed, n_states, n_actions, model, policy, gamma, lr):
		'''
		buffer_size : int -maximum memory size
		key : jax.random.PRNGKey
		n_states : int - dimensionality of state space
		n_actions : int - dimensionality of action space
		model : the q-network taking states to action values (hk.without_apply_rng(hk.transform))
		policy : exploration policy
		gamma : float - discount for future rewards
		lr : float - learning rate of mlp
		'''
		super(DQNAgent, self).__init__(keyseed, n_states, n_actions, gamma)

		self.buffer = ExperienceReplay(buffer_size)

		self.policy = policy

		self.target_params = self.params = model.init(next(self.prng), jnp.ones([1, n_states]))
		self.q_forward = model.apply # takes args : params, s

		opt_init, self.opt_update = optax.sgd(lr)
		self.opt_state = opt_init(self.params)

	def act(self, s, exploration = True):

		forward = lambda x: self.q_forward(self.params, x).squeeze()
		
		return self.policy(next(self.prng), s[None, :], self.n_actions, forward, exploration)

	def train(self, batch_size):

		if len(self.buffer) >= batch_size:

			# sample
			s, a, r, d, s_next = self.buffer.sample(batch_size) 

			# targets
			y = self.compute_target(r, d, s_next)
			
			# update parameters
			gradients = self.q_backward(self.params, s, y, a)

			updates, self.opt_state = self.opt_update(gradients, self.opt_state, self.params)
			self.params = optax.apply_updates(self.params, updates)

	@partial(jax.jit, static_argnums = 0)
	@partial(jax.grad, argnums  = 1)
	def q_backward(self, params, s, y, a):
		''' Computes the target loss for a given batch.'''
		q = self.q_forward(params, s)[jnp.arange(s.shape[0]), a]
		return jnp.mean((y - q) ** 2)

	@partial(jax.jit, static_argnums = 0)
	def compute_target(self, r, d, s):
		q_next = self.q_forward(self.target_params, s)
		return r + (1.0 - d) * self.gamma * jnp.max(q_next, axis = -1)

	def update_buffer(self, transition):
		self.buffer.update(transition)
	
	def update_target(self, tau  = 0.0):
		''' Updates the target network with parameters from the Q-network.
		tau : float = 0.0. The percentage of target network parameters to keep. If 0 then
		a hard update is done.
		'''
		self.target_params = jax.tree_util.tree_multimap(
			lambda x, y: tau * x + (1 - tau) * y,
			self.target_params,
			self.params)

class DDQNAgent(DQNAgent):
	''' A Double-DQN agent (https://arxiv.org/pdf/1509.06461.pdf). Only overwrites
	the compute_target method.'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@partial(jax.jit, static_argnums = 0)
	def compute_target(self, r, d, s):
		q_actions = jnp.argmax(self.q_forward(self.params, s), axis = -1)
		return r + (1.0 - d) * self.gamma * self.q_forward(self.target_params, s)[jnp.arange(s.shape[0]), q_actions]

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
	parser.add_argument('-eps_max', type = float, default = 1.0)
	parser.add_argument('-eps_decay', type = float, default = 8e-5)
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

	Agent = DDQNAgent

	keyseed = 4
	policy = EpsilonGreedy(lambda t: epsilon_min + epsilon_max * jnp.exp(-t * epsilon_decay))
	model = mlp.create_mlp([32, 32, 2])
	agent = Agent(buffer_size, keyseed, 4, 2, model, policy, gamma, lr)

	ep_rewards = []
	for episode in range(episodes):

		s = env.reset()
		d = False
		ep_reward = 0.0

		while not d:

			a = agent.act(jnp.array(s))

			s_next, r, d, _ = env.step(a)
			ep_reward += r

			agent.update_buffer(Transition(s = s, a = a, r = r, d = d, s_next = s_next))
			agent.train(batch_size)

			s = s_next

		ep_rewards.append(ep_reward)

		if episode % 100 == 0:
			agent.update_target()
			print('Episode {} Mean reward {}'.format(episode, np.mean(ep_rewards[-50:])))

	plt.plot(ep_rewards)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Episodic reward for {} on CartPole-v0.'.format(args.agent.upper()))
	plt.show()
