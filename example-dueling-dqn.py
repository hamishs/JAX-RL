'''
Implement a deuling architecture for dqn as in:
https://arxiv.org/pdf/1511.06581.pdf. Train an 
agent on CartPole-v0.
'''

import jax
import jax.numpy as jnp 
import numpy as np 
import haiku as hk 
import gym 
import matplotlib.pyplot as plt 
import argparse

import policies, dqn, utils

# parse args
parser = argparse.ArgumentParser(description = 'Deuling DQN on CartPole')
parser.add_argument('-eps_min', type = float, default = 0.01)
parser.add_argument('-eps_max', type = float, default = 0.15)
parser.add_argument('-eps_decay', type = float, default = 1e-5)
parser.add_argument('-buffer_size', type = int, default = 5000)
parser.add_argument('-gamma', type = float, default = 0.97)
parser.add_argument('-lr', type = float, default = 1e-3)
parser.add_argument('-episodes', type = int, default = 1500)
parser.add_argument('-batch_size', type = int, default = 32)
args = parser.parse_args()
epsilon_min = args.eps_min
epsilon_max = args.eps_max
epsilon_decay = args.eps_decay
buffer_size = args.buffer_size
gamma = args.gamma
lr = args.lr
episodes = args.episodes
batch_size = args.batch_size

'''
The deuling architecture computes the state-action value by
decomposing it into two functions: the value function (V(s))
and the advantage function (A(s,a)). The Q value is then
calculated as Q(s, a) = V(s) + A(s, a) - sum_a A(s, a) / N
where N is the number of actions.

To implement this the Q-network uses a base network to extract
features before splitting into two streams for each function which
are combined according to the equation above.
'''

def deuling_architecture(s):
	# base mlp
	base = hk.nets.MLP([32, 32], activate_final = True)

	# advantage and value functions
	A = hk.nets.MLP([16, 5])
	V = hk.nets.MLP([16, 1])

	# forward operations:
	x = base(s) # (batch, 32)
	a = A(x) # (batch, 5)
	v = V(x) # (batch, 1)

	# compute Q
	return v + v - v.mean(-1, keepdims = True)

model = hk.transform(deuling_architecture)

key = jax.random.PRNGKey(0)
policy = policies.EpsilonGreedy(lambda t: epsilon_min + epsilon_max * jnp.exp(-t * epsilon_decay))
agent = dqn.DDQNAgent(buffer_size, key, 4, 2, model, policy, gamma, lr)

env = gym.make('CartPole-v0')

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
		transition = utils.Transition(s = s, a = a, r = r, d = d, s_next = s_next)

		agent.train(transition, batch_size) # key = None as model is not random

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
