import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from abc import abstractmethod

from jax_rl.algorithms import BaseAgent

class TabularAlgorithm(BaseAgent):
	'''
	A base class for a tabular learning algorithm. Algorithms
	implement the train and q_targets method. q_targets computes
	the targets for the q-update. train implements this update.
	'''

	def __init__(self, key, n_states, n_actions, gamma, policy, lr):
		'''
		policy : a jax_rl.policies.Policy returning the next action. The behavioural policy.
		target_policy : returns the action used for the q-targets.
		'''
		super(TabularAlgorithm, self).__init__(key, n_states, n_actions, gamma)

		self.policy = policy
		self.lr = lr

		# initialise q-values
		self.q = jnp.zeros((n_states, n_actions))

	def act(self, s, exploration = True):
		return self.policy(next(self.prng), self.q[s,:], exploration = exploration)

	def train_on_env(self, env, episodes, verbose = None):
		'''Trains the q-targets on a given environment.'''
		rewards = []
		for epsiode in range(episodes):
			s = env.reset()
			d = False
			ep_reward = 0.0
			while not d:
				a = self.act(s, exploration = True)
				s_next, r, d, _ = env.step(a)
				ep_reward += r
				self.train(s, a, r, d, s_next)
				s = s_next
			rewards.append(ep_reward)

			if verbose is not None:
				if episode % verbose:
					print('Episode {} Reward {:.4f}'.format(epsiode, np.mean(rewards[-verbose:])))

		return rewards

	def train(self, s, a, r, d, s_next):
		'''Update on a previous transition.'''
		self.q = jax.ops.index_add(self.q, (s,a), self.lr * (r + self.gamma * self.q_targets(s_next) - self.q[s,a]))

	@abstractmethod
	def q_targets(self):
		'''Calcualte the Q-targets.'''
		pass

class QLearning(TabularAlgorithm):
	''' Implements Q-learning: off-policy model-free rl.'''
	def __init__(self, key, n_states, n_actions, gamma, policy, lr):
		super(QLearning, self).__init__(key, n_states, n_actions, gamma, policy, lr)

	def q_targets(self, s):
		return jnp.max(self.q[s])

class DoubleQLearning(TabularAlgorithm):
	''' Double Q-learning (Hasselt et al.)'''
	def __init__(self, key, n_states, n_actions, gamma, policy, lr):
		super(DoubleQLearning, self).__init__(key, n_states, n_actions, gamma, policy, lr)
		self.q2 = self.q.copy()

	@property
	def q_values(self):
		return 0.5 * (self.q + self.q2)
	
	def act(self, s, exploration = True):
		return self.policy(next(self.prng), self.q_values[s,:], exploration = exploration)

	def train(self, s, a, r, d, s_next):
		'''Update on a previous transition.'''
		if jax.random.uniform(next(self.prng)) > 0.5:
			self.q = jax.ops.index_add(self.q, (s,a), self.lr * (r + self.gamma * self.q_targets(s_next, other = True) - self.q[s,a]))
		else:
			self.q2 = jax.ops.index_add(self.q2, (s,a), self.lr * (r + self.gamma * self.q_targets(s_next, other = False) - self.q2[s,a]))

	def q_targets(self, s, other = False):
		if other:
			return jnp.max(self.q[s])
		else:
			return jnp.max(self.q2[s])

class SARSA(TabularAlgorithm):
	''' State-Action-Reward-State-Action.'''
	def __init__(self, key, n_states, n_actions, gamma, policy, lr):
		super(SARSA, self).__init__(key, n_states, n_actions, gamma, policy, lr)

	def q_targets(self, s):
		a = self.act(s, exploration = True)
		return self.q[s,a]

class ExpectedSARSA(TabularAlgorithm):
	''' SARSA with expectation over the next action. The policy passed needs
	to have a return_distribution keyword to return to return the distribution
	over the actions.'''
	def __init__(self, key, n_states, n_actions, gamma, policy, lr):
		super(ExpectedSARSA, self).__init__(key, n_states, n_actions, gamma, policy, lr)

	def act(self, s, exploration = True, return_distribution = False):
		return self.policy(next(self.prng), self.q[s,:], exploration = exploration,
			return_distribution = return_distribution)

	def q_targets(self, s):
		a_dist = self.act(s, exploration = True, return_distribution = True)
		return self.q[s,:].dot(a_dist)
		

	

