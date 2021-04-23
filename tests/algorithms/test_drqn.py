'''
Test PPO algorithm by training for 20 epidoes. 
'''

import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

import jax_rl
from jax_rl.algorithms import DRQN
from jax_rl.policies import EpsilonGreedy
from jax_rl.utils import lstm_initial_state

import gym
env = gym.make('CartPole-v0')

# Example LSTM network
def forward(s, hidden = None):
	'''
	Apply the LSTM over the input sequence with given initial state.
	s : (batch, seq_len, features)
	hidden : LSTM state (h, c).
	'''

	# LSTM
	lstm = hk.LSTM(16)
	if hidden is None: hidden = lstm.initial_state(s.shape[0])
	s, hidden = hk.dynamic_unroll(lstm, jnp.transpose(s, (1, 0, 2)), hidden)

	# output fully connected
	mlp2 = hk.nets.MLP([16, 1])
	s = hk.BatchApply(mlp2)(jnp.transpose(s, (1, 0, 2)))

	return s, hidden

model = hk.without_apply_rng(hk.transform(forward))
init_state = lambda batch_size: lstm_initial_state(16, batch_size = batch_size)

policy = EpsilonGreedy(lambda t : t ** -0.4)

drqn = DRQN(0, 4, 2, 0.99, 100, 200, policy, model, init_state, 1e-3)
ep_rewards, losses = drqn.train_on_env(env, 20, 1, verbose = 1)