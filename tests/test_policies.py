import jax
import jax.numpy as jnp 
import haiku as hk
from jax_rl.policies import EpsilonGreedy, BoltzmannPolicy

keys = hk.PRNGSequence(11)

# epsilon greedy
eps = EpsilonGreedy(0.1)

a = eps(next(keys), jax.random.normal(next(keys), (10,)), exploration = True)
assert a >= 0 and a <= 9

a = eps(next(keys), jax.random.normal(next(keys), (10,)), exploration = False)
assert a >= 0 and a <= 9
dist = eps(next(keys), jax.random.normal(next(keys), (10,)), return_distibution = True)

assert dist.shape == (10,)
assert dist.sum() > 0.999999

# boltzman
boltz = BoltzmannPolicy(1.0)

a = boltz(next(keys), jax.random.normal(next(keys), (10,)), exploration = True)
assert a >= 0 and a <= 9

a = boltz(next(keys), jax.random.normal(next(keys), (10,)), exploration = False)
assert a >= 0 and a <= 9

dist = boltz(next(keys), jax.random.normal(next(keys), (10,)), return_distibution = True)
assert dist.shape == (10,)
assert dist.sum() > 0.999999
