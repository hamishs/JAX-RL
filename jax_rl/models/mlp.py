'''
All models implemented in Haiku.
'''

import jax 
import jax.numpy as jnp 
import numpy as np 
import haiku as hk

def create_mlp(hidden, activation = jax.nn.relu):
	''' Creates a multi-layer perceptron model with given
	hidden sizes (list of int) and activations.'''

	def model(s):
		mlp = hk.nets.MLP(hidden, activation = jax.nn.relu)
		return mlp(s)

	return hk.transform(model)

if __name__ == '__main__':

	model = create_mlp([2, 5, 3])

	key = jax.random.PRNGKey(0)
	s = jnp.array([[0.0, 0.5], [-0.9, 2.5]])

	params = model.init(key, s)

	print(params)
	print(model.apply(params, None, s))