import numpy as np

class Simulator:
	def __init__ (self):
		dt = 0.1

		self.A = np.asarray([[1, dt], [0, 1]])
		self.B = np.asarray([[0], [dt]])

		self.range_q = np.asarray([1, 1]).reshape((2,1))
		self.kill_range_q = np.asarray([1, 1]).reshape((2,1)) * 2


		self.obs_dim = self.A.shape[1]
		self.act_dim = self.B.shape[1]

	def linear_step (self, q, act):
		u = 2*np.maximum(np.minimum(act, 1), 0).reshape((act.flatten().shape[0], 1)) - 1
		return self.A @ q + self.B @ u, u

	def step (self, q, dq, act):
		nq, u = self.linear_step(q, act)
		nq = nq + dq
		return nq, u

	def calc_rew (self, q, u, nq):
		return -np.mean(np.square(nq/self.kill_range_q)) - np.mean(np.square(u)) if not self.calc_done(nq) else -2
		
	def calc_done (self, q):
		return np.any(q > self.kill_range_q) or np.any(q < -self.kill_range_q)
	
	def reset (self, debug=False):
		if debug:
			q = np.asarray([0, 1]).reshape(self.range_q.shape)
		else:
			q = np.random.uniform(-self.range_q, self.range_q)
		return q
		
		