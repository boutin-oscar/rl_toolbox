import numpy as np

class Simulator:
	def __init__ (self):
		dt = 0.1

		A = np.asarray([[1, dt], [0, 1]])
		B = np.asarray([[0], [dt]])

		range_q = np.asarray([1, 1]).reshape((2,1))
		kill_range_q = np.asarray([1, 1]).reshape((2,1)) * 2


		obs_dim = A.shape[1]
		act_dim = B.shape[1]

		obs_mean = np.zeros(obs_dim)
		obs_std = np.ones(obs_dim)

		def linear_step (q, act):
			u = 2*np.maximum(np.minimum(act, 1), 0).reshape((act.flatten().shape[0], 1)) - 1
			return A @ q + B @ u, u

		def step (q, dq, act):
			nq, u = linear_step(q, act)
			nq = nq + dq
			return nq, u

		def reset (debug=False):
			if debug:
				q = np.asarray([0, 1]).reshape(range_q.shape)
			else:
				q = np.random.uniform(-range_q, range_q)
			return q