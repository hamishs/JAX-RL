import jax
from jax_rl.tabular import QLearning, DoubleQLearning, SARSA, ExpectedSARSA
from jax_rl.policies import EpsilonGreedy

q_learning = QLearning(4, 5, 10, 0.99, EpsilonGreedy(0.1), 0.1)
double_q = DoubleQLearning(4, 5, 10, 0.99, EpsilonGreedy(0.1), 0.1)
sarsa = SARSA(4, 5, 10, 0.99, EpsilonGreedy(0.1), 0.1)

class AltEpsilonGreedy:
	''' Epsilon greedy that returns distributions for expected SARSA.'''

	def __init__(self, epsilon):
		self.epsilon = epsilon

	def __call__(self, key, n_actions, q_values, exploration = True, return_distribution = False):
		if return_distribution:
			dist = jnp.ones(n_actions,) * self.epsilon / n_actions
			dist[jnp.argmax(q_values)] += 1 - self.epsilon
			return dist 
		else:
			if exploration and (jax.random.uniform(key) > 1 - self.epsilon):
				return jax.random.choice(key, n_actions)
			else:
				return int(jnp.argmax(q_values))

expected_sarsa = ExpectedSARSA(4, 5, 10, 0.99, AltEpsilonGreedy(0.1), 0.1)