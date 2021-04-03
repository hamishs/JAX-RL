import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

from functools import partial

from jax_rl.utils import Transition
from jax_rl.buffer import ExperienceReplay
from jax_rl.algorithms import BaseAgent 

class DDPG(BaseAgent):
	''' Deep Deterministic Policy Gradient for discrete action spaces.'''

	def __init__(self, key, n_states, n_actions, gamma, a_bounds, noise,
		buffer_size, actor, critic, lr_actor, lr_critic, tau):
		super(PPO, self).__init__(key, n_states, n_actions, gamma)

		self.a_low, self.a_high = a_bounds
		self.noise = noise

		self.buffer = TrajectoryBuffer(buffer_size)

		# initiate networks and optimisers
		self.actor_params = actor.init(next(self.prng), jnp.ones([1, n_states]))
		self.actor = actor.apply
		self.actor_update, self.actor_opt_state = self.init_optimiser(-lr_actor, self.actor_params)
		self.actor_target_params = hk.data_structures.to_immutable_dict(self.actor_params)

		self.critic_params = critic.init(next(self.prng), jnp.ones([1, n_states]), jnp.ones([1, n_actions]))
		self.critic = value.apply
		self.critic_update, self.critic_opt_state = self.init_optimiser(lr_critic, self.critic_params)
		self.critic_target_params = hk.data_structures.to_immutable_dict(self.critic_params)

		self.tau = tau

	def act(self, s, exploration = True):
		# s : (1, n_states)
		a = self.policy(self.policy_params, s).squeeze()
		if exploration:
			n_ = self.noise(next(self.prng))
			a = jnp.clip(a + n_, self.a_low, self.a_high)
		return a

	def train(self, batch_size):
		''' Train the algorithm on the current buffer.'''

		if len(self.buffer) > batch_size:
			s, a, r, d, s_next = self.buffer.sample(batch_size)

			# loss and gradients
			actor_loss, actor_grads = jax.value_and_grad(self.actor_loss)(self.actor_params, self.critic_params, s)
			critic_loss, critic_grads = jax.value_and_grad(self.critic_loss)(self.critic_params, s, a, r, d, s_next)
			
			# apply gradients
			a_updates, self.actor_opt_state = self.actor_update(
				actor_grads, self.actor_opt_state, self.actor_params)
			self.actor_params = optax.apply_updates(self.actor_params, a_updates)

			c_updates, self.critic_opt_state = self.critic_update(
				critic_grads, self.critic_opt_state, self.critic_params)
			self.critic_params = optax.apply_updates(self.critic_params, c_updates)	

			# update targets
			self.soft_update_targets(self.tau)	

			return actor_loss, critic_loss

	@partial(jax.jit, static_argnums = 0)
	def actor_loss(self, actor_params, critic_params, s):
		return jnp.mean(self.critic(self.ciritc_params, s, self.actor(self.actor_params, s)))

	@partial(jax.jit, static_argnums = 0)
	def critic_loss(self, critic_params, s, a, r, d, s_next):
		a_targ = self.actor(self.actor_target_params, s_next)
		y = r + self.gamma * (1 - d) * self.critic(self.critic_target_params, s_next, a_targ)
		return jnp.square(self.critic(critic_params, s, a) - y).mean()

	def update_buffer(self, s, a, r, d, s_next, pi):
		self.buffer.update(s, a, r, d, s_next, pi)
		
	def soft_update_targets(self, tau):
		''' Update targets with 1 - tau percent of current parameters.'''
		_update = lambda x, y : tau * x + (1.0 - tau) * y
		self.actor_target_params = jax.tree_util.tree_multimap(_update, self.actor_target_params, self.actor_params)
		self.critic_target_params = jax.tree_util.tree_multimap(_update, self.critic_target_params, self.critic_params)

	def init_optimiser(self, lr, params):
		opt_init, opt_update = optax.adam(lr)
		opt_state = opt_init(params)
		return opt_update, opt_state

	def train_on_env(self, env, episodes, batch_size, verbose = None):
		''' Train on a given environment.
		env : environment with methods reset and step e.g. gym CartPole-v1
		episodes : number of training episodes
		update_freq : frequency of training episodes
		train_steps : number of gradient descent steps each training epsiode
		verbose : wether to print current rewards. Given as int refering to print frequency.
		'''
		ep_rewards = []
		actor_losses, critic_losses = [], []

		for episode in range(episodes):
			s = env.reset()
			d = False
			ep_reward = 0.0
			while not d:
				a = self.act(jnp.array([s]))
				s_next, r, d, _ = env.step(a)
				ep_reward += r
				self.update_buffer(Transition(s=s, a=a, r=r, d=d, s_next=s_next))
				a_loss, c_loss = self.train(batch_size)
				if a_loss is not None: actor_losses.append(a_loss)
				if c_loss is not None: critic_losses.append(c_loss)
				s = s_next
				ep_length += 1

			# ep end
			ep_rewards.append(ep_reward)

			if verbose is not None:
				if episode % verbose == 0:
					print('Episode {} Reward {:.4f} Actor Loss {:.4f} Critic Loss {:.4f}'.format(episode,
						np.mean(ep_rewards[-verbose:]),
						np.mean(actor_losses[-verbose:]),
						np.mean(critic_losses[-verbose:])))

		return ep_rewards, actor_losses, critic_losses

if __name__ == '__main__':

	import gym
	env = gym.make('MountainCarContinuous-v0')
	
	# actor mu(s) -> a
	def forward(s):
		mlp = hk.nets.MLP([32, 32, 1])
		return jax.nn.tanh(mlp)
	actor = hk.without_apply_rng(hk.transform(forward))

	# critic Q(s, a) -> q
	def forward(s, a):
		x = jnp.concatenate((s, a), axis = 1) # (batch, n_states + n_actions)
		mlp = hk.nets.MLP([32, 32, 2])
		return mlp(x)
	critic = hk.without_apply_rng(hk.transform(forward))

	from utils import GaussianNoise
	noise = GaussianNoise(sd = 0.05)

	ddpq = DDPG(5, 2, 1, 0.98, (-1.0, 1.0), noise, 5000, actor, critic, 1e-4, 1e-3, 0.95)
	ep_rewards, actor_losses, critic_losses = ddpg.train_on_env(env, 1500, 64, verbose = 10)

