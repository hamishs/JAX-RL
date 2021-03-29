from jax_rl.policies import EpsilonGreedy, BoltzmannPolicy

eps = EpsilonGreedy(0.1)
boltz = BoltzmannPolicy(1.0)
