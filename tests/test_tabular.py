import jax
from jax_rl.tabular import QLearning, DoubleQLearning, SARSA, ExpectedSARSA
from jax_rl.policies import EpsilonGreedy

q_learning = QLearning(4, 5, 10, 0.99, EpsilonGreedy(0.1), 0.1)
double_q = DoubleQLearning(4, 5, 10, 0.99, EpsilonGreedy(0.1), 0.1)
sarsa = SARSA(4, 5, 10, 0.99, EpsilonGreedy(0.1), 0.1)
expected_sarsa = ExpectedSARSA(4, 5, 10, 0.99, EpsilonGreedy(0.1), 0.1)