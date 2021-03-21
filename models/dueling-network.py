import jax 
import jax.numpy as jnp 
import numpy as np 
import haiku as hk

def create_dueling_network(base_hidden, A_hidden, V_hidden):
	'''
	Dueling network for DQN. Initial MLP breaks into
	two streams to produce a value (V(s)) and advantage
	function (A(s, a)). These are combined to produce the
	state-value estimate, Q(s, a), by:
	Q(s, a) = V(s) + A(s, a) - sum_a A(s, a) / N where
	N is the number of actions.
	'''

	def forward(s):
		base = hk.nets.MLP(base_hidden, activate_final = True)
		A = hk.nets.MLP(A_hidden)
		V = hk.nets.MLP(V_hidden)

		x = base(s)
		a = A(x) # (batch, n_actions)
		v = V(x) # (batch, 1)

		return v + a - a.mean(-1, keepdims = True)

	return hk.transform(forward)
