# JAX-RL
JAX implementations of various deep reinforcement learning algorithms.

Main libraries used:
* JAX - main framework
* Haiku - neural networks
* optax - gradient based optimisation

Algorithms:
* Deep Q-Network (DQN)
* Double Deep Q-Network (DDQN)
* Deep Deterministic Policy Gradient (DDPG)

Policies:
* Epsilon-greedy
* Boltzmann

TODO:
* Prioritised experience replay
* Multi-agent DQN and DDPG
* other models e.g. CNN.
* Meta-gradient RL?

Example: train DQN on CartPole from Open-AI gym:
```
python3 dqn.py -gamma 0.97 -episodes 750 
```
