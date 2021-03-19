'''
For work in progress:

Currently: PrioritizedExperiencerReplay with DQN.
'''


class PrioritizedExperienceReplay:
	'''Experience replay buffer using prioritised selection.'''

	def __init__(self, buffer_size):
		'''
		buffer_size : int - max number of transitions to store.
		'''
		self.buffer_size = buffer_size 
		self.buffer = [] 
		self.priorities = []
	
	def update(self, transition):
		''' Update the buffer with a transition and maximal priority.'''
		self.buffer.append(transition)

		p = jnp.max(jnp.array(priorities))
		self.priorities.append(p)
		if len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)
			self.priorities.pop(0)
	
	def __len__(self):
		return len(self.buffer)
	
	def sample(self, key, n, alpha = 0.0):
		''' Sample a batch of n transitions according to a priority
		distribution with hyperparameter a. a = 0 gives uniform distirbution
		large a gives greedy choice. Returns all (state, action, reward, done,
		next state, priorities) as jax.numpy arrays.'''

		p = jnp.array(self.priorities) ** alpha
		p /= p.sum()

		key, subkey = jax.random.split(key)
		idxs = jax.random.choice(subkey, len(self.buffer), shape = (n,), p = p)

		batch = Transition(*zip(*[self.buffer[i] for i in idxs]))
		
		s = jnp.array(batch.s, dtype = jnp.float32)
		a = jnp.array(batch.a, dtype = jnp.int32)
		r = jnp.array(batch.r, dtype = jnp.float32)
		d = jnp.array(batch.d, dtype = jnp.float32)
		s_next = jnp.array(batch.s_next, dtype = jnp.float32)
		ps = p[idxs]

		return s, a, r, d, s_next, p


class PrioritisedDQNAgent:

	def __init__(self, buffer_size, key, n_states, n_actions, hidden, policy, gamma, lr, K, alpha, beta):
		'''
		buffer_size : int -maximum memory size
		key : jax.random.PRNGKey
		n_states : int - dimensionality of state space
		n_actions : int - dimensionality of action space
		hidden : list int -  hidden sizes for mlp
		policy : exploration policy
		gamma : float - discount for future rewards
		lr : float - learning rate of mlp
		K : int - frequency of priority updates
		k : int - number of priority updates to perform
		alpha : float - parameter for priority distribution
		beta : float - parameter for importance sampling weight
		'''

		# environment parameters
		self.n_states = n_states
		self.n_actions = n_actions

		#  prioritised experience replay buffer
		self.buffer = PrioritizedExperienceReplay(buffer_size)
		self.K = K
		self.alpha = alpha 
		self.beta = beta
		self._t = 0

		# exploration policy
		self.policy = policy

		# future reward discount
		self.gamma = gamma

		# Q-network and update functions
		key, subkey = jax.random.split(key)
		self.params = mlp.mlp_init_fn([n_states] + hidden + [n_actions], subkey)
		self.target_params = self.params.copy()
		self.q_forward = mlp.batch_forward
		self.q_backward = jax.jit(jax.grad(lambda params, s, a: mlp.batch_forward(params, s)[jnp.arange(s.shape[0]), a]))
		self.optimizer = mlp.get_optimizer(lr = lr)

	def act(self, key, s, exploration = True):
		key, subkey = jax.random.split(key)

		s = jnp.array(s)[jnp.newaxis, :]
		forward = lambda x: self.q_forward(self.params, x).squeeze()

		return self.policy(subkey, s, self.n_actions, forward, exploration)
	
	def train(self, key, transition, batch_size):

		self.buffer.update(transition)

		if (self._t % self.K != 0) or (len(self.buffer) < batch_size):
			return self.params
	
		# sample and compute importance-sampling weights
		s, a, r, d, s_next, p = self.buffer.sample(key, batch_size, alpha = self.alpha) 
		w = (len(self.buffer) * p) ** -self.beta
		w /= w.sum()

		# td error
		a_q_next = jnp.argmax(self.q_forward(self.params, s_next), axis = -1)
		q_next_target = self.q_forward(self.target_params, s_next)[jnp.arange(s.shape[0]), a_q_next]
		delta = r + (1 - d) * (self.gamma * q_next_target - self.q_forward(self.params, s)[jnp.arange(s.shape[0]), a])
		
		# update priorities + compute gradients
		p = jnp.abs(delta)
		gradients = w * delta * self.q_backward(self.params, s, a)

		# apply gradients
		self.params = self.optimizer(self.params, gradients)
		self._t += 1
	
	def update_target(self):
		self.target_params = self.params.copy()