import setuptools

setuptools.setup(
	name = 'jax_rl',
	version = '0.1.0',
	author = 'Hamish Scott',
	author_email = 'hamish.scott@icloud.com',
	packages=setuptools.find_packages(where="src"),
	package_dir={'': 'src'},
	url = 'https://github.com/hamishs/JAX-RL',
	license = 'LICENSE',
	description = 'JAX implementations of deep reinforcement learning algorithms.',
	install_requires=["numpy >= 1.19.5", "jax >= 0.2.10", "dm-haiku", "optax"],
	)