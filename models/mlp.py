import jax 
import jax.numpy as jnp 
import numpy as np 

#init function
def mlp_init_fn(h, key):
	''' Creates a MLP with network sizes h.
	h : list of int
	key : a jax.random.PRNGKey key.
	'''

	keys = jax.random.split(key, len(h))

	def init_layer(m, n, key, scale = 1e-2):
		w_key, b_key = jax.random.split(key)
		w = scale * jax.random.normal(w_key, (n, m))
		b = scale * jax.random.normal(b_key, (n,))
		return w, b
	
	return [init_layer(m, n, k) for m, n, k in zip(h[:-1], h[1:], keys)]

# forward function
def forward(params, x):
	''' Forward function for a MLP with RELU connections and softmax output.
	params : list of tuples of jax arrays with parameters for each layer.'''

	for w, b in params[:-1]:
		x = jax.nn.relu(jnp.dot(w, x) + b)
	
	w, b = params[-1]
	return jnp.dot(w, x) + b

# create a batched version of the forward function
batch_forward = jax.jit(jax.vmap(forward, in_axes = (None, 0), out_axes = 0))

def get_optimizer(lr = 1e-3):
	''' Returns a jax.jit function that applies one step of
	stochastic gradient descent to given parameters with
	given gradients using the learning rate.
	lr : float.
	'''

	def sgd(params, gradients):
		'''
		params : list of tuples of network parametesr.
		gradients : list of tuples of same shapes as params with gradients.
		'''
		new_params = []
		for (w, b), (dw, db) in zip(params, gradients):
			new_params.append((w - lr * dw, b - lr * db))
		return new_params
	
	return jax.jit(sgd)