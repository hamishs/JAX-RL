# JAX-RL
JAX implementations of various reinforcement learning algorithms.

Main libraries used:
* JAX - main framework
* Haiku - neural networks in JAX
* optax - gradient based optimisation in JAX

Algorithms:
* Deep Q-Network (DQN)

Policies:
* Epsilon-greedy
* Boltzmann

TODO:
* Prioritised experience replay
* DDPG
* Multi-agent DQN and DDPG
* other models e.g. CNN.

Example: train DQN on CartPole from Open-AI gym:
```
python3 dqn.py -gamma 0.97 -episodes 750 
```
