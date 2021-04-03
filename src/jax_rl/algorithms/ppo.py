import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

from functools import partial

from jax_rl.algorithms import BaseAgent 
from jax_rl.buffer import TrajectoryBuffer
from jax_rl.utils import entropy

class PPO(BaseAgent):
	''' Proximal Policy Optimisation for discrete action spaces.'''

	def __init__(self, key, n_states, n_actions, gamma, lmda, epsilon,
		v_weight, e_weight, buffer_size, policy, value, lr_policy, lr_value):
		super(PPO, self).__init__(key, n_states, n_actions, gamma)

		self.lmda = lmda
		self.epsilon = epsilon
		self.v_weight = v_weight
		self.e_weight = e_weight

		self.buffer = TrajectoryBuffer(buffer_size)

		# initiate networks and optimisers
		self.policy_params = policy.init(next(self.prng), jnp.ones([1, n_states]))
		self.policy = policy.apply
		self.policy_update, self.policy_opt_state = self.init_optimiser(lr_policy, self.policy_params)

		self.value_params = value.init(next(self.prng), jnp.ones([1, n_states]))
		self.value = value.apply
		self.value_update, self.value_opt_state = self.init_optimiser(lr_value, self.value_params)

	def act(self, s, return_prob = False):
		# s : (1, n_states)
		p = self.policy(self.policy_params, s).squeeze()
		a = int(jax.random.choice(next(self.prng), self.n_actions, p = p))
		if return_prob:
			return a, p[a]
		else:
			return a

	def train(self):
		''' Train the algorithm on the current buffer.'''
		s, a, r, d, s_next, pi_old, advs, tds = self.buffer.data
		advs = advs.squeeze()

		# loss and gradients
		loss, (policy_grads, value_grads) = jax.value_and_grad(self.loss_fn, argnums = (0, 1))(
			self.policy_params, self.value_params, s, a, r, d, s_next, pi_old, advs, tds)
		
		# apply gradients
		p_updates, self.policy_opt_state = self.policy_update(
			policy_grads, self.policy_opt_state, self.policy_params)
		self.policy_params = optax.apply_updates(self.policy_params, p_updates)

		v_updates, self.value_opt_state = self.value_update(
			value_grads, self.value_opt_state, self.value_params)
		self.value_params = optax.apply_updates(self.value_params, v_updates)		

		return loss

	@partial(jax.jit, static_argnums = 0)
	def loss_fn(self, policy_params, value_params, s, a, r, d, s_next, pi_old, advs, tds):
		# actor loss
		pi_new = self.policy(policy_params, s)
		ratio = jnp.exp(jnp.log(pi_new[jnp.arange(a.shape[0]), a.squeeze()]) - jnp.log(pi_old))
		actor_loss = jnp.minimum(ratio * advs, jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advs) 
		
		# entropy
		e = entropy(pi_new, axis = -1)
		
		# critic loss
		v_loss = 0.5 * jnp.square((self.value(value_params, s) - tds).squeeze())
		
		# compute loss and update
		loss = (-actor_loss + self.v_weight * v_loss - self.e_weight * e)
		return loss.mean()

	def update_buffer(self, s, a, r, d, s_next, pi):
		self.buffer.update(s, a, r, d, s_next, pi)
		
	def compute_advantages(self, l):
		''' Compute advantage estimates for the previous l steps of the buffer.'''
		s = jnp.array(self.buffer.state[-l:]) # (l, n_states)
		r = jnp.array(self.buffer.reward[-l:]) # (l, 1)
		s_next = jnp.array(self.buffer.state_next[-l:]) # (l, n_states)
		d = jnp.array(self.buffer.done[-l:]) # (l, 1)

		td_target = r + self.gamma * self.value(self.value_params, s_next) * (1 - d)
		delta = (td_target - self.value(self.value_params, s)).squeeze() # (l,)
		
		# calculate advantage
		advantages = []
		adv = 0.0
		for d in delta[::-1]:
			adv = self.gamma * self.lmda * adv + d
			advantages.append([adv])
		advantages.reverse() # (l, 1)
		
		# update tds
		if self.buffer.tds.shape == (0,):
			self.buffer.tds = td_target
		else:
			self.buffer.tds = jnp.vstack((self.buffer.tds, td_target))
		self.buffer.advs += advantages

	def init_optimiser(self, lr, params):
		opt_init, opt_update = optax.adam(lr)
		opt_state = opt_init(params)
		return opt_update, opt_state

	def train_on_env(self, env, episodes, update_freq, train_steps, verbose = None):
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
			ep_length = 0
			ep_reward = 0.0
			while not d:
				a, pi = self.act(jnp.array([s]), return_prob = True)
				s_next, r, d, _ = env.step(a)
				ep_reward += r
				self.update_buffer(s, a, r, d, s_next, pi)
				s = s_next
				ep_length += 1

			# ep end
			self.compute_advantages(ep_length)
			ep_rewards.append(ep_reward)
			if episode % update_freq == 0:
				for _ in range(train_steps):
					loss = self.train()
					losses.append(loss)

			if verbose is not None:
				if episode % verbose == 0:
					print('Episode {} Reward {:.4f} Loss {:.4f}'.format(episode,
						np.mean(ep_rewards[-verbose:]),
						np.mean(losses[-verbose*train_steps:])))

		return ep_rewards, losses


if __name__ == '__main__':

	import gym
	env = gym.make('CartPole-v0')
	
	def forward(s):
		mlp = hk.nets.MLP([16, 32, 2])
		return jax.nn.softmax(mlp(s))
	policy = hk.without_apply_rng(hk.transform(forward))

	def forward(s):
		mlp = hk.nets.MLP([16, 32, 1])
		return mlp(s)
	value = hk.without_apply_rng(hk.transform(forward))

	ppo = PPO(0, 4, 2, 0.99, 0.95, 0.2, 0.9, 0.01, 400, policy, value, 1e-2, 1e-2)
	ep_rewards, losses = ppo.train_on_env(env, 500, 1, 3, verbose = 10)

