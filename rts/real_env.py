import numpy as np
import rts.simulator as sim

class RealEnv: # we usually do not have access to this.
	def __init__ (self, debug=False):
		
		self.range_q = sim.range_q
		self.kill_range_q = sim.kill_range_q

		
		self.obs_dim = sim.obs_dim
		self.act_dim = sim.act_dim
		
		self.obs_mean = sim.obs_mean
		self.obs_std = sim.obs_std
		
		self.num_envs = 1
		
		self.debug = debug
		
		self.max_f = 0.01
	
	def step (self, act):
		lin_s, u = sim.linear_step (self.q, act)
		dq = np.asarray([0, -lin_s[1,0]]).reshape(lin_s.shape)
		dq = np.maximum(np.minimum(dq, self.max_f), -self.max_f)
		self.dq = dq
		self.q, u = sim.step(self.q, dq, act)
		
		rew = -np.mean(np.square(self.q/self.kill_range_q)) - np.mean(np.square(u))
		rew = rew if not self.calc_done() else -2
		
		return self.calc_obs(), [rew], [self.calc_done()]
	
	def calc_obs (self):
		return [self.q.flatten()]
	
	def calc_rew (self):
		return -np.mean(np.square(self.q/self.kill_range_q)) if not self.calc_done() else -1
	
	def calc_done (self):
		return np.any(self.q > self.kill_range_q) or np.any(self.q < -self.kill_range_q)
	
	def reset (self):
		self.q = np.random.uniform(-self.range_q, self.range_q)
		self.q = np.asarray([0, 1]).reshape(self.q.shape)
		if self.debug:
			self.q = np.asarray([0, 1]).reshape(self.q.shape)
		
		return self.calc_obs ()