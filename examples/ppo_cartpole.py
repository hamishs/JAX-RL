try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("This example requires matplotlib")

import gym
import haiku as hk
import jax

from jax_rl.algorithms import PPO

env = gym.make("CartPole-v0")


# define MLPs for the value function and policy
def forward(s):
    mlp = hk.nets.MLP([16, 32, 2])
    return jax.nn.softmax(mlp(s))


policy = hk.without_apply_rng(hk.transform(forward))


def forward(s):
    mlp = hk.nets.MLP([16, 32, 1])
    return mlp(s)


value = hk.without_apply_rng(hk.transform(forward))

seed = 2  # JAX random seed

# env parameters
n_states = 4
n_actions = 2

# hyperparameters
gamma = 0.99  # discount rate
lmda = 0.95  # advantage decay
epsilon = 0.2  # max clip
v_weight = 0.9  # value loss weight
e_weight = 0.01  # entropy loss weight
buffer_size = 450  # max number of transitions to store

# training parameters
lr_policy = 9e-3
lr_value = 9e-3
episodes = 180
train_freq = 1
train_steps = 3

# define the algorithm and train on CartPole
ppo = PPO(
    seed,
    n_states,
    n_actions,
    gamma,
    lmda,
    epsilon,
    v_weight,
    e_weight,
    buffer_size,
    policy,
    value,
    lr_policy,
    lr_value,
)
ep_rewards, losses = ppo.train_on_env(env, episodes, train_freq, train_steps, verbose=1)

plt.plot(ep_rewards)
plt.title("Episodic reward of PPO on CartPole")
plt.show()
