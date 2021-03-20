'''
Deep Deterministic Policy Gradient (DDPG) as presented in
Continuous control with deep reinforcement learning
(Lillicrap et al., 2016)
https://arxiv.org/pdf/1509.02971
'''

import jax
import jax.numpy as jnp 
import numpy as np 
import haiku as hk 
import optax

from utils import Transition, ExperienceReplay

class DDPGAgent:
	''' A Deep Deterministic Policy Gradient agent.'''

	def __init__(self, buffer_size, key, n_states, n_actions, actor,
		critic, noise, gamma, tau, lr_actor, lr_critic):
		'''
		buffer_size : int -maximum memory size
		key : jax.random.PRNGKey
		n_states : int - dimensionality of state space
		n_actions : int - dimensionality of action space
		critic : the critic network taking states and actions to q-values
		actor : the actor network taking states to action preferences
		noise : callable stochastic process for exploration noise
		tau : float - target networks update rate
		lr_actor: float - learning rate of the actor network
		lr_critic : float - learning rate of critic network
		'''

		self.buffer = ExperienceReplay(buffer_size, action_dtype = jnp.float32)

		self.n_states = n_states
		self.n_actions = n_actions

		# initialise model parameters
		key, subkey = jax.random.split(key)
		self.actor_params = actor.init(subkey, jnp.ones([1, n_states]))
		self.actor_target_params = hk.data_structures.to_immutable_dict(self.actor_params)
		self.actor_forward = actor.apply # params, key, s -> a

		self.critic_params = critic.init(subkey, jnp.ones([1, n_states]), jnp.ones([1, n_actions]))
		self.critic_target_params = hk.data_structures.to_immutable_dict(self.critic_params)
		self.critic_forward = critic.apply # params, key, s, a -> R

		self.noise = noise
		self.gamma = gamma
		self.tau = tau

		def critic_loss(params, key, s, y, a):
			''' Computes the target loss for a given batch.'''
			q = self.critic_forward(params, key, s, a)
			return jnp.mean((y - q) ** 2)
		self.critic_backward = jax.jit(jax.grad(critic_loss))

		opt_init, self.critic_opt_update = optax.sgd(lr_critic)
		self.critic_opt_state = opt_init(self.critic_params)

		def actor_loss(params, key, s, critic_params):
			a = self.actor_forward(params, key, s)
			q = self.critic_forward(critic_params, key, s, a)
			return jnp.mean(q)
		self.actor_backward = jax.jit(jax.grad(actor_loss))

		opt_init, self.actor_opt_update = optax.sgd(-lr_actor) # gradient ascent
		self.actor_opt_state = opt_init(self.actor_params)

	def act(self, key, s, exploration = True):
		''' Sample an action with or without exploration noise.'''
		key, subkey = jax.random.split(key)

		s = jnp.array(s)[jnp.newaxis, :]
		a = self.actor_forward(self.actor_params, subkey, s).squeeze()

		return a + self.noise(subkey) if exploration else a
	
	def train(self, transition, batch_size, key):
		''' Train the agent for one step after observing a given transition.'''

		key, subkey = jax.random.split(key)

		self.buffer.update(transition)

		if len(self.buffer) >= batch_size:
			
			s, a, r, d, s_next = self.buffer.sample(batch_size) 

			# update critic
			target_actions = self.actor_forward(self.actor_target_params, subkey, s_next)
			y = r + self.gamma * self.critic_forward(self.critic_target_params, subkey, s_next, target_actions)

			critic_grads = self.critic_backward(self.critic_params, subkey, s, y, a[:, jnp.newaxis])
			updates, self.critic_opt_state = self.critic_opt_update(critic_grads, self.critic_opt_state, self.critic_params)
			self.critic_params = optax.apply_updates(self.critic_params, updates)

			# update actor
			actor_grads = self.actor_backward(self.actor_params, subkey, s, self.critic_params)
			updates, self.actor_opt_state = self.actor_opt_update(actor_grads, self.actor_opt_state, self.actor_params)
			self.actor_params = optax.apply_updates(self.actor_params, updates)

			self.update_target_critic()
			self.update_target_actor()

	def update_target_critic(self):
		self.critic_target_params = jax.tree_util.tree_multimap(
			lambda x, y: self.tau * x + (1 - self.tau) * y,
			self.critic_params,
			self.critic_target_params)

	def update_target_actor(self):
		self.actor_target_params = jax.tree_util.tree_multimap(
			lambda x, y: self.tau * x + (1 - self.tau) * y,
			self.actor_params,
			self.actor_target_params)
		
if __name__ == '__main__':

	from utils import GaussianNoise
	from models import mlp 

	import matplotlib.pyplot as plt

	import gym
	env = gym.make('MountainCarContinuous-v0')

	buffer_size = 10000
	n_states = 2
	n_actions = 1
	noise = GaussianNoise(sd = lambda t: 0.005 + 0.1 * np.exp(-t * 0.0001))
	gamma = 0.97
	tau = 0.1
	lr_critic = 1e-3
	lr_actor = 1e-4

	batch_size = 32
	max_steps = 250
	episodes = 300

	def actor_model(s):
		mlp = hk.nets.MLP([32, 32, 1], activation = jax.nn.relu)
		return jax.nn.hard_tanh(mlp(s))
	actor = hk.transform(actor_model)

	def critic_model(s, a):
		# s : (batch, 1)
		# a : (batch, 1)

		s1 = hk.Linear(16)
		a1 = hk.Linear(16)
		hidden1 = hk.Linear(32)
		final = hk.Linear(1)

		x1 = jax.nn.relu(s1(s)) # (batch, 16)
		x2 = jax.nn.relu(a1(a)) # (batch, 16)
		return final(jax.nn.relu(hidden1(jnp.concatenate([x1, x2], axis = 1)))) # (batch, 1)
	critic = hk.transform(critic_model)

	key = jax.random.PRNGKey(0)
	agent = DDPGAgent(buffer_size, key, n_states, n_actions, actor, critic,
		noise, gamma, tau, lr_actor, lr_critic)

	ep_rewards = []
	for episode in range(episodes):

		ep_reward = 0.0
		d = False
		s = env.reset()

		for _ in range(max_steps):

			key, subkey = jax.random.split(key)

			# take action
			a = agent.act(subkey, s)
			s_next, r, d, _ = env.step((a,))
			ep_reward += r

			# train agent
			transition = Transition(s = s, a = a, r = r, d = d, s_next = s_next)
			agent.train(transition, batch_size, subkey)

			s = s_next

			if d:
				break

		ep_rewards.append(ep_reward)

		if episode % 2 == 0:
			print('Episode {} Mean reward {:.4f}'.format(episode, np.mean(ep_rewards[-2:])))

	plt.plot(ep_rewards)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Episodic reward on Mountain Car.')
	plt.show()



		





