'''
Test PPO algorithm by training for 20 epidoes. 
'''

import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

import jax_rl
from jax_rl.algorithms import DQN, DDQN
from jax_rl.policies import EpsilonGreedy

import gym
env = gym.make('CartPole-v0')

policy = EpsilonGreedy(0.1)

def forward(s):
	mlp = hk.nets.MLP([32, 32, 2])
	return mlp(s)
model = hk.without_apply_rng(hk.transform(forward))

# test DQN
dqn = DQN(0, 4, 2, 0.99, 500, policy, model, 1e-3)
ep_rewards, losses = dqn.train_on_env(env, 10, 32, 20, verbose = 1)

# test DDQN
ddqn = DDQN(0, 4, 2, 0.99, 500, policy, model, 1e-3)
ep_rewards, losses = ddqn.train_on_env(env, 10, 32, 20, verbose = 1)