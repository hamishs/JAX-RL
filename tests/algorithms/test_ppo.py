"""
Test PPO algorithm by training for 20 epidoes.
"""

import gym
import haiku as hk
import jax

from jax_rl.algorithms import PPO

env = gym.make("CartPole-v0")


def forward(s):
    mlp = hk.nets.MLP([16, 32, 2])
    return jax.nn.softmax(mlp(s))


policy = hk.without_apply_rng(hk.transform(forward))


def forward(s):
    mlp = hk.nets.MLP([16, 32, 1])
    return mlp(s)


value = hk.without_apply_rng(hk.transform(forward))

ppo = PPO(0, 4, 2, 0.99, 0.95, 0.2, 0.9, 0.01, 400, policy, value, 1e-2, 1e-2)
ep_rewards, losses = ppo.train_on_env(env, 20, 1, 3, verbose=1)
